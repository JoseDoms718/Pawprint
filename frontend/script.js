const fileUpload = document.getElementById("fileUpload");
const preview = document.getElementById("preview");

// Change this if your server runs elsewhere
const PREDICT_URL = "http://localhost:5000/predict";
const GENERATE_PDF_URL = "http://localhost:5000/generate_pdf";

let uploadedFile = null; // keep uploaded file for PDF

// Utility: fade + translate + scale animation
function animateElement(el, { opacity = [0, 1], translateY = [20, 0], translateX = [0, 0], scale = [1, 1], duration = 600, delay = 0 } = {}) {
  let start = null;
  function step(timestamp) {
    if (!start) start = timestamp;
    const elapsed = timestamp - start - delay;
    if (elapsed < 0) {
      requestAnimationFrame(step);
      return;
    }
    const progress = Math.min(elapsed / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    const currentOpacity = opacity[0] + (opacity[1] - opacity[0]) * eased;
    const currentY = translateY[0] + (translateY[1] - translateY[0]) * eased;
    const currentX = translateX[0] + (translateX[1] - translateX[0]) * eased;
    const currentScale = scale[0] + (scale[1] - scale[0]) * eased;
    el.style.opacity = currentOpacity;
    el.style.transform = `translate(${currentX}px, ${currentY}px) scale(${currentScale})`;
    if (progress < 1) requestAnimationFrame(step);
  }
  el.style.opacity = opacity[0];
  el.style.transform = `translate(${translateX[0]}px, ${translateY[0]}px) scale(${scale[0]})`;
  requestAnimationFrame(step);
}

// Progress circle animation
function animateProgress(target, circle, text, duration = 1500) {
  const circumference = 490;
  let start = null;
  function step(timestamp) {
    if (!start) start = timestamp;
    const progress = Math.min((timestamp - start) / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    const current = Math.round(eased * target);
    text.textContent = current + "%";
    const offset = circumference - (current / 100) * circumference;
    circle.style.strokeDashoffset = offset;
    if (progress < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

// Circular loader animation
function animateLoaderCircle(circle) {
  const circumference = 220;
  let angle = 0;
  function spin() {
    angle += 4;
    const offset = circumference - ((angle % 360) / 360) * circumference;
    circle.style.strokeDashoffset = offset;
    requestAnimationFrame(spin);
  }
  requestAnimationFrame(spin);
}

// File upload listener
fileUpload.addEventListener("change", async function () {
  const file = this.files[0];
  if (!file) return;

  uploadedFile = file; // store file for PDF later

  // Show loading
  preview.innerHTML = `
    <div class="w-full flex flex-col items-center mt-16" id="loadingBox">
      <p class="text-xl text-[#7a3f1a] mb-8">Uploading and predicting...</p>
      <div class="relative w-[100px] h-[100px] flex items-center justify-center">
        <svg class="absolute w-full h-full -rotate-90">
          <circle cx="50%" cy="50%" r="35" stroke="#eee" stroke-width="10" fill="none"/>
          <circle id="loadingCircle" cx="50%" cy="50%" r="35"
            stroke="#e26215" stroke-width="10" fill="none"
            stroke-linecap="round" stroke-dasharray="220" stroke-dashoffset="220"/>
        </svg>
      </div>
    </div>
  `;

  const loadingBox = document.getElementById("loadingBox");
  const loadingCircle = document.getElementById("loadingCircle");
  animateElement(loadingBox, { opacity: [0, 1], translateY: [30, 0], duration: 600 });
  animateLoaderCircle(loadingCircle);

  const form = new FormData();
  form.append("image", file);

  try {
    const res = await fetch(PREDICT_URL, { method: "POST", body: form });

    if (!res.ok) {
      const text = await res.text();
      preview.innerHTML = `<p class="text-red-600">Server error: ${text}</p>`;
      return;
    }

    const data = await res.json();
    if (data.error) {
      preview.innerHTML = `<p class="text-red-600">Error: ${data.error}</p>`;
      return;
    }

    const top = data.predictions[0];
    const exampleImg = top.example_image || "static/placeholder.png";
    const confidence = Math.round(top.confidence * 100);

    // Build result UI (only static example image)
    preview.innerHTML = `
      <div class="flex flex-col md:flex-row items-center justify-center gap-8 md:gap-16 w-full max-w-[1000px] mt-12 mx-auto" id="resultBox">
        <div class="flex flex-col items-center text-center gap-6 max-w-[450px]">
          <div class="relative w-[170px] h-[170px] flex items-center justify-center">
            <svg class="absolute w-full h-full -rotate-90">
              <circle cx="50%" cy="50%" r="78" stroke="#ddd" stroke-width="15" fill="none"/>
              <circle id="progressCircle" cx="50%" cy="50%" r="78"
                stroke="#e26215" stroke-width="15" fill="none"
                stroke-linecap="round" stroke-dasharray="490" stroke-dashoffset="490"/>
            </svg>
            <span id="progressText" class="text-4xl font-bold text-[#7a3f1a] leading-none">0%</span>
          </div>
          <div class="flex flex-col items-center gap-3">
            <h3 id="breedName" class="font-poppins font-extrabold text-[26px] text-[#e26215]">
              ${top.breed.replace(/_/g, " ").toUpperCase()}
            </h3>
            <p id="breedDesc" class="italic text-[#555] text-[16px] leading-[1.6] text-center max-w-[420px]">
              Prediction confidence: ${confidence}%. This is the model's best guess based on visual features.
            </p>
          </div>
        </div>
        <div class="flex flex-col items-center gap-5">
          <img id="breedImage" src="${exampleImg}" alt="Example of predicted breed"
            class="w-[350px] h-[220px] object-cover rounded-xl shadow-md border-2 border-[#e26215]" />
          <div class="flex gap-3">
            <a id="downloadReport" href="#"
              class="btn-ghost font-poppins font-bold text-[18px] px-8 py-3 rounded-full border-4 border-[#e26215] bg-transparent text-[#e26215] no-underline inline-block transition-all duration-300 hover:-translate-y-1 hover:bg-[#e26215] hover:text-white hover:border-[#cc500f]">
              PRINT REPORT
            </a>
          </div>
        </div>
      </div>
    `;

    // Animate
    animateElement(document.getElementById("resultBox"), { opacity: [0, 1], translateY: [40, 0], duration: 700 });
    animateElement(document.getElementById("breedImage"), { opacity: [0, 1], scale: [0.9, 1], duration: 800, delay: 200 });
    animateProgress(confidence, document.getElementById("progressCircle"), document.getElementById("progressText"), 1500);

    // PDF generation
    const downloadBtn = document.getElementById("downloadReport");
    downloadBtn.addEventListener("click", async (e) => {
      e.preventDefault();
      downloadBtn.disabled = true;
      downloadBtn.textContent = "GENERATING...";

      const formData = new FormData();
      formData.append("breed", top.breed);
      formData.append("confidence", top.confidence);
      if (uploadedFile) formData.append("image", uploadedFile);

      try {
        const res = await fetch(GENERATE_PDF_URL, { method: "POST", body: formData });
        if (!res.ok) throw new Error(`Server returned status ${res.status}`);
        const data = await res.json();
        const fileName = "PawPrint_Report_" + top.breed.replace(/_/g, "") + ".pdf";

        if (data.pdf_url) {
          downloadBtn.href = data.pdf_url;
          downloadBtn.setAttribute("download", fileName);
          window.open(data.pdf_url, "_blank");
          downloadBtn.textContent = "PRINT REPORT";
        } else {
          alert("Failed to generate PDF report: PDF URL missing from response.");
          downloadBtn.textContent = "PRINT REPORT";
        }
      } catch (err) {
        console.error("PDF generation failed:", err);
        alert(`Error generating PDF report: ${err.message || "Server error"}`);
        downloadBtn.textContent = "PRINT REPORT";
      } finally {
        downloadBtn.disabled = false;
      }
    });
  } catch (err) {
    preview.innerHTML = `<p class="text-red-600">Network error: ${err.message}</p>`;
  }
});
