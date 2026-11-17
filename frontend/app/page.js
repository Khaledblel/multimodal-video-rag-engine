"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { API, clock, getJson, streamAnswer, uploadVideo } from "./lib/api";

const ACTIVE = new Set(["queued", "running"]);
const HEALTH = {
  checking: "Checking the API",
  ok: "API ready",
  degraded: "API degraded",
  down: "API unreachable",
};

/** The ingest jobs that wait, run or failed, with the stages of the one that runs. */
function Jobs({ jobs }) {
  const shown = jobs.filter((job) => job.status !== "done").slice(0, 5);
  if (shown.length === 0) return null;
  return (
    <ul className="jobs">
      {shown.map((job) => {
        const at = job.stages.indexOf(job.stage);
        return (
          <li key={job.id} className={`job ${job.status}`}>
            <div className="row">
              <span className="name">{job.source_name}</span>
              <span className="status">{job.status}</span>
            </div>
            {job.status === "running" && (
              <ol className="stages">
                {job.stages.map((stage, i) => (
                  <li key={stage} className={i === at ? "now" : i < at ? "past" : ""}>
                    {stage.replace("_", " ")}
                  </li>
                ))}
              </ol>
            )}
            {job.error && <pre className="error">{job.error}</pre>}
          </li>
        );
      })}
    </ul>
  );
}

/** The answer's text, with each citation such as [2] turned into a button that plays that moment. */
function Answer({ text, moments, onCite }) {
  const known = new Set(moments.map((moment) => moment.n));
  // Splitting on a pattern in parentheses keeps the citations, as parts of their own.
  const parts = text.split(/(\[[\d,\s]+\])/g);
  return (
    <p className="answer">
      {parts.map((part, i) => {
        if (!/^\[[\d,\s]+\]$/.test(part)) return <span key={i}>{part}</span>;
        const numbers = (part.match(/\d+/g) || []).map(Number);
        return (
          <span key={i}>
            {numbers.map((n, j) =>
              known.has(n) ? (
                <button
                  type="button"
                  className="cite"
                  key={`${n}-${j}`}
                  onClick={() => onCite(n)}
                  title="Play this moment"
                >
                  {n}
                </button>
              ) : (
                <span key={`${n}-${j}`}>[{n}]</span>
              ),
            )}
          </span>
        );
      })}
    </p>
  );
}

/** The moments the answer was written from: where each is, the frames the model saw, and what is said. */
function Moments({ moments, cited, onPlay }) {
  const used = new Set(cited);
  return (
    <ol className="moments">
      {moments.map((moment) => {
        const name = moment.source_name || moment.video_id;
        return (
          <li key={moment.n} className={used.has(moment.n) ? "moment cited" : "moment"}>
            <button type="button" className="head" onClick={() => onPlay(moment.video_id, name, moment.start)}>
              <span className="number">{moment.n}</span>
              <span className="name">{name}</span>
              <span className="times">
                {clock(moment.start)} to {clock(moment.end)}
              </span>
            </button>
            {moment.frames.length > 0 && (
              <div className="frames">
                {moment.frames.map((frame) => (
                  <button
                    type="button"
                    key={frame.index}
                    onClick={() => onPlay(moment.video_id, name, frame.t)}
                    title={`Play from ${clock(frame.t)}`}
                  >
                    <img
                      src={`${API}/videos/${moment.video_id}/frames/${frame.index}?w=240`}
                      alt={`Frame at ${clock(frame.t)}`}
                      loading="lazy"
                    />
                    <span>{clock(frame.t)}</span>
                  </button>
                ))}
              </div>
            )}
            {moment.lines.length > 0 && (
              <ul className="lines">
                {moment.lines.map((line, i) => (
                  <li key={i}>
                    <button
                      type="button"
                      className="time"
                      onClick={() => onPlay(moment.video_id, name, line.start)}
                    >
                      {clock(line.start)}
                    </button>
                    <span>{line.text}</span>
                  </li>
                ))}
              </ul>
            )}
          </li>
        );
      })}
    </ol>
  );
}

/** The video player. `playing` names a video and a time; each new value loads the video if needed and seeks. */
function Player({ playing }) {
  const element = useRef(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    const video = element.current;
    if (!video || !playing) return undefined;
    setFailed(false);
    const seek = () => {
      video.currentTime = playing.time;
      // The jump follows a click, so the browser allows playing. If it still refuses, the video stays paused.
      video.play().catch(() => {});
    };
    if (video.dataset.videoId === playing.videoId && video.readyState >= 1) {
      seek();
      return undefined;
    }
    video.dataset.videoId = playing.videoId;
    video.addEventListener("loadedmetadata", seek, { once: true });
    video.src = `${API}/videos/${playing.videoId}/video`;
    video.load();
    return () => video.removeEventListener("loadedmetadata", seek);
  }, [playing]);

  return (
    <section className="panel player">
      <h2>{playing ? `${playing.name} at ${clock(playing.time)}` : "Player"}</h2>
      <video ref={element} controls preload="metadata" onError={() => setFailed(true)} />
      {!playing && <p className="hint">Pick a moment or a video, and it plays here.</p>}
      {failed && <p className="error">This video has no copy that the browser can play.</p>}
    </section>
  );
}

export default function Page() {
  const [health, setHealth] = useState("checking");
  const [videos, setVideos] = useState([]);
  const [jobs, setJobs] = useState([]);
  const [libraryError, setLibraryError] = useState("");
  const [upload, setUpload] = useState(null);

  const [question, setQuestion] = useState("");
  const [scope, setScope] = useState("");
  const [asking, setAsking] = useState(false);
  const [answer, setAnswer] = useState("");
  const [moments, setMoments] = useState([]);
  const [cited, setCited] = useState([]);
  const [plan, setPlan] = useState(null);
  const [stats, setStats] = useState(null);
  const [askError, setAskError] = useState("");

  const [playing, setPlaying] = useState(null);
  const running = useRef(null); // stops the answer that is being read when a new question is asked

  const refresh = useCallback(async () => {
    try {
      const [nextVideos, nextJobs] = await Promise.all([getJson("/videos"), getJson("/jobs")]);
      setVideos(nextVideos);
      setJobs(nextJobs);
      setLibraryError("");
    } catch (error) {
      setLibraryError(String(error.message || error));
    }
  }, []);

  const busy = upload !== null || jobs.some((job) => ACTIVE.has(job.status));

  useEffect(() => {
    refresh();
    if (!busy) return undefined;
    const timer = setInterval(refresh, 2000);
    return () => clearInterval(timer);
  }, [busy, refresh]);

  useEffect(() => {
    let alive = true;
    fetch(`${API}/health`)
      .then((response) => {
        if (alive) setHealth(response.ok ? "ok" : "degraded");
      })
      .catch(() => {
        if (alive) setHealth("down");
      });
    return () => {
      alive = false;
    };
  }, []);

  function play(videoId, name, time) {
    setPlaying({ videoId, name, time, at: Date.now() });
  }

  function cite(n) {
    const moment = moments.find((item) => item.n === n);
    if (moment) play(moment.video_id, moment.source_name || moment.video_id, moment.start);
  }

  function videoName(videoId) {
    const video = videos.find((item) => item.video_id === videoId);
    return (video && video.source_name) || videoId;
  }

  async function onFile(event) {
    const file = event.target.files && event.target.files[0];
    event.target.value = "";
    if (!file) return;
    setLibraryError("");
    setUpload({ name: file.name, fraction: 0 });
    try {
      await uploadVideo(file, (fraction) => setUpload({ name: file.name, fraction }));
    } catch (error) {
      setLibraryError(`${file.name}: ${error.message || error}`);
    } finally {
      setUpload(null);
      refresh();
    }
  }

  async function onAsk(event) {
    event.preventDefault();
    const q = question.trim();
    if (!q) return;
    if (running.current) running.current.abort();
    const controller = new AbortController();
    running.current = controller;
    setAsking(true);
    setAnswer("");
    setMoments([]);
    setCited([]);
    setPlan(null);
    setStats(null);
    setAskError("");
    try {
      await streamAnswer(
        scope ? { q, video_id: scope } : { q },
        (item) => {
          if (item.type === "moments") {
            setMoments(item.moments);
            setPlan(item.plan);
          } else if (item.type === "token") {
            setAnswer((text) => text + item.text);
          } else if (item.type === "done") {
            setAnswer(item.answer);
            setCited(item.citations);
            setStats(item.stats);
          } else if (item.type === "error") {
            setAskError(item.detail);
          }
        },
        controller.signal,
      );
    } catch (error) {
      if (error.name !== "AbortError") setAskError(String(error.message || error));
    } finally {
      if (running.current === controller) {
        running.current = null;
        setAsking(false);
      }
    }
  }

  return (
    <main className="page">
      <header className="top">
        <h1>Video RAG</h1>
        <span className={`health ${health}`}>{HEALTH[health]}</span>
      </header>

      <div className="columns">
        <div className="column">
          <Player playing={playing} />

          <section className="panel">
            <h2>Videos</h2>
            <label className="upload">
              <input type="file" accept="video/*,.mkv" onChange={onFile} disabled={upload !== null} />
              <span>
                {upload ? `Sending ${upload.name}: ${Math.round(upload.fraction * 100)}%` : "Add a video"}
              </span>
            </label>
            {upload && <progress value={upload.fraction} max="1" />}
            {libraryError && <p className="error">{libraryError}</p>}
            <Jobs jobs={jobs} />
            <ul className="videos">
              {videos.map((video) => (
                <li key={video.video_id}>
                  <button
                    type="button"
                    disabled={!video.playable}
                    onClick={() => play(video.video_id, video.source_name || video.video_id, 0)}
                    title={video.playable ? "Play from the start" : "No copy for the browser yet"}
                  >
                    {video.source_name || video.video_id}
                  </button>
                  <span className="meta">
                    {clock(video.duration_s)}
                    {video.indexed ? "" : ", not indexed"}
                    {video.playable ? "" : ", not playable"}
                  </span>
                </li>
              ))}
            </ul>
            {videos.length === 0 && !busy && <p className="hint">No video yet. Add one to begin.</p>}
          </section>
        </div>

        <div className="column">
          <section className="panel ask">
            <h2>Ask</h2>
            <form onSubmit={onAsk}>
              <input
                type="text"
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                placeholder="Ask about your videos"
                maxLength={500}
                aria-label="Question"
              />
              <select value={scope} onChange={(event) => setScope(event.target.value)} aria-label="Videos to search">
                <option value="">Every video</option>
                {videos
                  .filter((video) => video.indexed)
                  .map((video) => (
                    <option key={video.video_id} value={video.video_id}>
                      {video.source_name || video.video_id}
                    </option>
                  ))}
              </select>
              <button type="submit" disabled={!question.trim()}>
                Ask
              </button>
            </form>

            {askError && <p className="error">{askError}</p>}
            {asking && !answer && (
              <p className="hint">{moments.length > 0 ? "Reading the moments..." : "Searching..."}</p>
            )}
            {answer && <Answer text={answer} moments={moments} onCite={cite} />}
            {plan && plan.planned && plan.video_id && (
              <p className="hint">Searched in {videoName(plan.video_id)} only, because the question names it.</p>
            )}
            {stats && (
              <p className="stats">
                {stats.moments} moments, {stats.frames} frames, {stats.prompt_tokens ?? "?"} prompt tokens,{" "}
                {stats.total_s.toFixed(1)} s
              </p>
            )}
            {moments.length > 0 && <Moments moments={moments} cited={cited} onPlay={play} />}
          </section>
        </div>
      </div>
    </main>
  );
}
