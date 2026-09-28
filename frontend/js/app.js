"use strict";

const form = document.querySelector("#translate-form");
const input = document.querySelector("#sentence-input");
const translateButton = document.querySelector("#translate-button");
const statusMessage = document.querySelector("#status-message");
const resultPanel = document.querySelector("#result-panel");
const emptyState = document.querySelector("#empty-state");
const matchedSentence = document.querySelector("#matched-sentence");
const glossValue = document.querySelector("#gloss-value");
const videoCount = document.querySelector("#video-count");
const video = document.querySelector("#sign-video");
const videoPosition = document.querySelector("#video-position");
const previousButton = document.querySelector("#previous-video");
const nextButton = document.querySelector("#next-video");

let variants = [];
let currentVariant = 0;

function setStatus(message, kind = "") {
  statusMessage.textContent = message;
  statusMessage.className = `status ${kind}`.trim();
}

function setLoading(loading) {
  translateButton.disabled = loading;
  translateButton.classList.toggle("is-loading", loading);
  translateButton.querySelector(".button-label").textContent = loading ? "Searching…" : "Translate to ISL";
  form.setAttribute("aria-busy", String(loading));
}

function clearResult() {
  variants = [];
  currentVariant = 0;
  video.pause();
  video.removeAttribute("src");
  video.load();
  resultPanel.hidden = true;
  emptyState.hidden = false;
}

function showUnknown() {
  clearResult();
  emptyState.querySelector("p").textContent = "No matching ISL video found for this sentence.";
  setStatus("No exact sentence match was found.");
}

function renderVariant() {
  if (!variants.length) return;
  const item = variants[currentVariant];
  video.pause();
  video.src = item.video_url;
  video.load();
  videoPosition.textContent = `Video ${currentVariant + 1} of ${variants.length}`;
  previousButton.disabled = currentVariant === 0;
  nextButton.disabled = currentVariant === variants.length - 1;
}

function showMatch(result) {
  variants = result.videos;
  currentVariant = 0;
  matchedSentence.textContent = result.sentence || result.sentence_normalized;
  glossValue.textContent = result.gloss;
  videoCount.textContent = String(result.video_count);
  emptyState.hidden = true;
  resultPanel.hidden = false;
  renderVariant();
  setStatus("Matching sentence found.", "success");
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = input.value.trim();
  if (!text) {
    clearResult();
    emptyState.querySelector("p").textContent = "Your matching sign video will appear here.";
    setStatus("Enter a sentence to search.", "error");
    input.focus();
    return;
  }

  setStatus("");
  setLoading(true);
  try {
    const response = await fetch("/sign/translate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (!response.ok) {
      throw new Error(response.status === 422 ? "Please enter a valid sentence." : "The sign service is unavailable right now.");
    }
    const result = await response.json();
    if (result.matched) {
      showMatch(result);
    } else {
      showUnknown();
    }
  } catch (error) {
    clearResult();
    emptyState.querySelector("p").textContent = "Your matching sign video will appear here.";
    setStatus(error instanceof TypeError ? "Could not connect to the sign service. Please try again." : error.message, "error");
  } finally {
    setLoading(false);
  }
});

previousButton.addEventListener("click", () => {
  if (currentVariant > 0) {
    currentVariant -= 1;
    renderVariant();
  }
});

nextButton.addEventListener("click", () => {
  if (currentVariant < variants.length - 1) {
    currentVariant += 1;
    renderVariant();
  }
});
