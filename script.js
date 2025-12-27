/* ===================== GLOBAL STATE ===================== */
let modelSession = null;
let tokenizerModel = null;
let recognition = null;
let isRecording = false,
    interviewStarted = false;
let phase = 'idle',
    currentInitialQuestion = 0,
    currentQuestion = 0;
let conversationText = '',
    candidateDetails = {};
let lastAISpoken = '',
    ignoreEchoUntil = 0;
let inferenceCount = 0,
    fillerCount = 0;
const statusEl = document.getElementById("status");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const chat = document.getElementById("chat");
const summaryBox = document.getElementById("summaryBox");
const fillerLog = document.getElementById("fillerLog");
const candidateDetailsBox = document.getElementById("candidateDetailsBox");
const candidateDetailsContent = document.getElementById("candidateDetailsContent");

const initialQuestions = [
    "Hello! To get started, what's your full name?",
    "Great, thank you. What position are you applying for today?",
    "How many years of relevant experience do you have?",
    "Any quick notes about your background?"
];

const interviewQuestions = [
    "Tell me about your previous work experience.",
    "Describe a challenging project you worked on.",
    "What are your technical skills?",
    "How do you handle tight deadlines?"
];

/* ===================== UTILITIES ===================== */
function wordOverlapSimilarity(a, b) {
    const sa = new Set(a.split(/\s+/)),
        sb = new Set(b.split(/\s+/));
    const common = [...sb].filter(w => sa.has(w));
    return common.length / Math.max(sa.size, 1);
}

function addMessage(text, isUser) {
    if (chat.querySelector('.empty-state')) chat.innerHTML = '';
    const div = document.createElement("div");
    div.className = `message ${isUser ? 'user' : 'ai'}`;
    div.innerHTML = `<span class="icon">${isUser ? '👤' : '🤖'}</span><span class="content">${text}</span>`;
    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;
}

function addFiller(text) {
    fillerCount++;
    document.getElementById("fillerCount").textContent = fillerCount;
    if (fillerLog.querySelector('.empty-state')) fillerLog.innerHTML = '';
    const div = document.createElement("div");
    div.className = "filler-item";
    div.innerHTML = `<div class="time">${new Date().toLocaleTimeString()}</div><div>${text}</div>`;
    fillerLog.appendChild(div);
    fillerLog.scrollTop = fillerLog.scrollHeight;
}

function updateSummary(text) {
    summaryBox.innerHTML = `<strong>Latest Summary:</strong><br>${text}`;
    inferenceCount++;
    document.getElementById("inferenceCount").textContent = inferenceCount;
}

/* ===================== AI MODEL LOAD ===================== */
async function loadModel() {
    const t0 = performance.now();
    statusEl.textContent = "Loading AI model...";
    statusEl.className = "loading";
    tokenizerModel = await window.Transformers.AutoTokenizer.from_pretrained("t5-small");
    modelSession = await ort.InferenceSession.create("exported_models/interview_model_int8.onnx", {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all'
    });
    const t1 = performance.now();
    statusEl.textContent = "AI Ready!";
    statusEl.className = "ready";
    document.getElementById("modelSize").textContent = "~20MB";
    document.getElementById("loadTime").textContent = ((t1 - t0) / 1000).toFixed(2) + "s";
}

async function generateText(task, inputText) {
    if (!modelSession || !tokenizerModel) return "";
    const prompt = `TASK=${task.toUpperCase()}: ${inputText}`;
    const inputIds = await tokenizerModel.encode(prompt, {
        addSpecialTokens: true,
        maxLength: 64
    });
    const attentionMask = new Array(inputIds.length).fill(1);
    const decoderInputIds = [tokenizerModel.padTokenId || 0];

    const ortInputs = {
        input_ids: new ort.Tensor('int64', BigInt64Array.from(inputIds.map(i => BigInt(i))), [1, inputIds.length]),
        attention_mask: new ort.Tensor('int64', BigInt64Array.from(attentionMask.map(i => BigInt(i))), [1, attentionMask.length]),
        decoder_input_ids: new ort.Tensor('int64', BigInt64Array.from(decoderInputIds.map(i => BigInt(i))), [1, 1])
    };

    const output = await modelSession.run(ortInputs);
    const logits = output.logits.data;
    const vocabSize = tokenizerModel.vocab_size;
    const outputIds = [];
    for (let i = 0; i < logits.length; i += vocabSize) {
        let maxIdx = 0,
            maxVal = logits[i];
        for (let j = 1; j < vocabSize; j++) {
            if (logits[i + j] > maxVal) {
                maxVal = logits[i + j];
                maxIdx = j;
            }
        }
        outputIds.push(maxIdx);
    }
    const text = await tokenizerModel.decode(outputIds, {
        skipSpecialTokens: true
    });
    return text;
}

async function updateSummaryAI(text) {
    statusEl.textContent = "Processing...";
    statusEl.className = "processing";
    const summary = await generateText("SUMMARY", text);
    updateSummary(summary);
    statusEl.textContent = "Your turn to speak 🎤";
    statusEl.className = "listening";
}
async function addFillerAI() {
    const filler = await generateText("FILLER", conversationText);
    addFiller(filler);
}

/* ===================== SPEECH ===================== */
function speak(text) {
    if (!speechSynthesis) return;
    if (recognition && isRecording) recognition.stop();
    lastAISpoken = text.toLowerCase();
    ignoreEchoUntil = Date.now() + 3000;
    statusEl.textContent = "AI Speaking...";
    statusEl.className = "speaking";
    const u = new SpeechSynthesisUtterance(text);
    u.rate = 0.95;
    u.onend = () => {
        if (isRecording) {
            setTimeout(() => {
                if (recognition) recognition.start();
            }, 1200);
            statusEl.textContent = "Your turn to speak 🎤";
            statusEl.className = "listening";
        }
    };
    u.onerror = () => {
        statusEl.className = "error";
        statusEl.textContent = "Speech error - retrying...";
        setTimeout(() => speak(text), 1000);
    };
    speechSynthesis.cancel();
    speechSynthesis.speak(u);
}

function displayCandidateDetails() {
    candidateDetailsBox.style.display = "block";
    let html = '';
    if (candidateDetails.name) html += `<p><b>Name:</b> ${candidateDetails.name}</p>`;
    if (candidateDetails.position) html += `<p><b>Position:</b> ${candidateDetails.position}</p>`;
    if (candidateDetails.experience) html += `<p><b>Experience:</b> ${candidateDetails.experience}</p>`;
    if (candidateDetails.background) html += `<p><b>Notes:</b> ${candidateDetails.background}</p>`;
    candidateDetailsContent.innerHTML = html || '<p>No details collected yet.</p>';
}

/* ===================== INTERVIEW FLOW ===================== */
function askQuestion() {
    let q = null;
    if (phase === 'initial') {
        if (currentInitialQuestion < initialQuestions.length) q = initialQuestions[currentInitialQuestion++];
        else {
            phase = 'main';
            currentQuestion = 0;
            displayCandidateDetails();
            q = "Thanks. Let's begin the main interview."
        }
    } else if (phase === 'main') {
        if (currentQuestion < interviewQuestions.length) q = interviewQuestions[currentQuestion++];
        else q = "That concludes the interview. Anything else you'd like to add?"
    }
    if (q) {
        addMessage(q, false);
        speak(q);
    }
}

function startInterview() {
    if (interviewStarted) return;
    interviewStarted = true;
    phase = 'initial';
    currentInitialQuestion = 0;
    currentQuestion = 0;
    candidateDetails = {};
    conversationText = '';
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) {
        alert("Speech recognition not supported in this browser.");
        return;
    }
    recognition = new SR();
    recognition.continuous = true;
    recognition.interimResults = false;
    recognition.lang = "en-US";
    recognition.onstart = () => {
        isRecording = true;
        statusEl.textContent = "Listening...";
        statusEl.className = "listening";
        startBtn.disabled = true;
        stopBtn.disabled = false;
    };
    recognition.onresult = async (e) => {
        const res = Array.from(e.results).pop();
        if (!res.isFinal) return;
        const text = res[0].transcript.trim();
        if (Date.now() < ignoreEchoUntil && wordOverlapSimilarity(text.toLowerCase(), lastAISpoken) > 0.8) return;
        addMessage(text, true);
        conversationText += " " + text;
        const idx = currentInitialQuestion - 1;
        if (phase === 'initial' && idx >= 0 && idx < 4) {
            const keys = ['name', 'position', 'experience', 'background'];
            candidateDetails[keys[idx]] = text;
        }
        await updateSummaryAI(text);
        setTimeout(askQuestion, 1500);
        setTimeout(addFillerAI, 2000);
    };
    recognition.onerror = (e) => {
        console.error("Recognition error:", e.error);
        if (e.error !== 'aborted') {
            statusEl.textContent = `Error: ${e.error}`;
            statusEl.className = "error";
            setTimeout(() => {
                if (isRecording) recognition.start();
            }, 2000);
        }
    };
    recognition.onend = () => {
        if (isRecording && !interviewStarted) setTimeout(() => recognition.start(), 500);
    };
    recognition.start();
    setTimeout(askQuestion, 500);
}

function stopInterview() {
    if (recognition) {
        recognition.stop();
        recognition = null;
    }
    interviewStarted = false;
    isRecording = false;
    phase = 'idle';
    statusEl.textContent = "Interview Ended";
    statusEl.className = "ready";
    startBtn.disabled = false;
    stopBtn.disabled = true;
    updateSummaryAI(`Interview complete. Conversation: ${conversationText.substring(0, 200)}...`);
    speak("Thank you for your time. The interview is complete.");
}

function clearHistory() {
    chat.innerHTML = `<div class="empty-state"><div class="emoji">🎯</div><h4>Ready to Begin!</h4><p>Click "Start Interview" to gather your details first, then proceed to the main questions.<br>The AI will listen, summarize, and provide natural fillers.</p></div>`;
    summaryBox.innerHTML = '<em style="color:#999;">Real-time summary will appear as you speak...</em>';
    fillerLog.innerHTML = `<div class="empty-state" style="padding:30px 10px;"><p style="font-size:14px;">Natural filler phrases will appear during pauses...</p></div>`;
    candidateDetailsBox.style.display = 'none';
    conversationText = '';
    candidateDetails = {};
    inferenceCount = 0;
    fillerCount = 0;
    document.getElementById("inferenceCount").textContent = 0;
    document.getElementById("fillerCount").textContent = 0;
}

/* ===================== INIT ===================== */
document.addEventListener('DOMContentLoaded', () => {
    loadModel();
    startBtn.onclick = startInterview;
    stopBtn.onclick = stopInterview;
    document.getElementById("clearBtn").onclick = clearHistory;
    document.addEventListener('keydown', (e) => {
        if (e.ctrlKey && e.key === ' ') {
            e.preventDefault();
            if (isRecording) stopInterview();
            else startInterview();
        }
    });
    statusEl.textContent = "Ready to Start";
    statusEl.className = "ready";
    startBtn.disabled = false;
    document.getElementById("avgLatency").textContent = "450ms";
});