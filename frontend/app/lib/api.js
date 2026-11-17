export const API = "/api";

/** 75.4 becomes "1:15" and 3725 becomes "1:02:05". */
export function clock(seconds) {
  const total = Math.max(0, Math.floor(seconds));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = String(total % 60).padStart(2, "0");
  return hours ? `${hours}:${String(minutes).padStart(2, "0")}:${secs}` : `${minutes}:${secs}`;
}

export async function reason(response) {
  try {
    const body = await response.json();
    if (typeof body.detail === "string") return body.detail;
    return JSON.stringify(body.detail ?? body);
  } catch {
    return `the server answered ${response.status}`;
  }
}

export async function getJson(path) {
  const response = await fetch(`${API}${path}`);
  if (!response.ok) throw new Error(await reason(response));
  return response.json();
}

/**
 * Send a file as the body of a PUT, the form in which the browser reports how far an upload is.
 * Resolves with the ingest job that the API queued.
 */
export function uploadVideo(file, onProgress) {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("PUT", `${API}/videos?name=${encodeURIComponent(file.name)}`);
    request.upload.onprogress = (event) => {
      if (event.lengthComputable) onProgress(event.loaded / event.total);
    };
    request.onload = () => {
      let body = null;
      try {
        body = JSON.parse(request.responseText);
      } catch {
        body = null;
      }
      if (request.status >= 200 && request.status < 300) {
        resolve(body);
      } else {
        const detail = body && typeof body.detail === "string" ? body.detail : `the server answered ${request.status}`;
        reject(new Error(detail));
      }
    };
    request.onerror = () => reject(new Error("the upload did not reach the server"));
    request.send(file);
  });
}

/**
 * Ask a question and hand each event of the streamed reply to onEvent.
 * The reply is one JSON object per line: "moments", then "token" pieces, then "done" or "error".
 */
export async function streamAnswer(params, onEvent, signal) {
  const query = new URLSearchParams({ ...params, stream: "true" });
  const response = await fetch(`${API}/ask?${query}`, { signal });
  if (!response.ok) throw new Error(await reason(response));

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let end = buffer.indexOf("\n");
    while (end >= 0) {
      const line = buffer.slice(0, end).trim();
      buffer = buffer.slice(end + 1);
      if (line) onEvent(JSON.parse(line));
      end = buffer.indexOf("\n");
    }
  }
  if (buffer.trim()) onEvent(JSON.parse(buffer));
}
