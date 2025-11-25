const fileUpload = document.getElementById("fileUpload");
const preview = document.getElementById("preview");

// Backend URLs: change PREDICT_URL if deploying too
const PREDICT_URL = "http://localhost:8000/predict"; // local dev
const GENERATE_PDF_URL = "http://localhost:8000/generate_pdf";

let uploadedFile = null;      // store uploaded file for PDF
let lastPrediction = null;    // store last prediction globally

// -------------------------
// Utility Animations
// -------------------------
function animateElement(el, { opacity=[0,1], translateY=[20,0], translateX=[0,0], scale=[1,1], duration=600, delay=0 }={}) {
    let start = null;
    function step(ts){
        if(!start) start = ts;
        const elapsed = ts - start - delay;
        if(elapsed < 0){ requestAnimationFrame(step); return; }
        const progress = Math.min(elapsed/duration,1);
        const eased = 1 - Math.pow(1 - progress, 3);
        el.style.opacity = opacity[0] + (opacity[1]-opacity[0])*eased;
        el.style.transform = `translate(${translateX[0] + (translateX[1]-translateX[0])*eased}px, ${translateY[0] + (translateY[1]-translateY[0])*eased}px) scale(${scale[0]+(scale[1]-scale[0])*eased})`;
        if(progress<1) requestAnimationFrame(step);
    }
    el.style.opacity = opacity[0];
    el.style.transform = `translate(${translateX[0]}px, ${translateY[0]}px) scale(${scale[0]})`;
    requestAnimationFrame(step);
}

function animateProgress(target, circle, text, duration=1500){
    const circumference = 490;
    let start = null;
    function step(ts){
        if(!start) start=ts;
        const progress = Math.min((ts-start)/duration,1);
        const eased = 1 - Math.pow(1 - progress,3);
        const current = Math.round(eased*target);
        text.textContent = current+"%";
        circle.style.strokeDashoffset = circumference - (current/100)*circumference;
        if(progress<1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
}

function animateLoaderCircle(circle){
    const circumference=220;
    let angle=0;
    function spin(){
        angle+=4;
        const offset = circumference - ((angle%360)/360)*circumference;
        circle.style.strokeDashoffset = offset;
        requestAnimationFrame(spin);
    }
    requestAnimationFrame(spin);
}

// -------------------------
// File upload + predict
// -------------------------
fileUpload.addEventListener("change", async function(){
    const file = this.files[0];
    if(!file) return;

    uploadedFile = file; // store for PDF

    // Show loading
    preview.innerHTML = `
        <div class="w-full flex flex-col items-center mt-16" id="loadingBox">
            <p class="text-xl text-[#7a3f1a] mb-8">Uploading and predicting...</p>
            <div class="relative w-[100px] h-[100px] flex items-center justify-center">
                <svg class="absolute w-full h-full -rotate-90">
                    <circle cx="50%" cy="50%" r="35" stroke="#eee" stroke-width="10" fill="none"/>
                    <circle id="loadingCircle" cx="50%" cy="50%" r="35" stroke="#e26215" stroke-width="10" fill="none" stroke-linecap="round" stroke-dasharray="220" stroke-dashoffset="220"/>
                </svg>
            </div>
        </div>
    `;
    const loadingBox = document.getElementById("loadingBox");
    const loadingCircle = document.getElementById("loadingCircle");
    animateElement(loadingBox, {opacity:[0,1], translateY:[30,0], duration:600});
    animateLoaderCircle(loadingCircle);

    const form = new FormData();
    form.append("image", file);

    try{
        const res = await fetch(PREDICT_URL, { method:"POST", body:form });
        if(!res.ok) throw new Error(`Server returned ${res.status}`);
        const data = await res.json();
        if(data.error) throw new Error(data.error);

        lastPrediction = data.predictions[0]; // store globally
        const top = lastPrediction;
        const exampleImg = top.example_image || "static/placeholder.png";
        const confidence = Math.round(top.confidence*100);

        // Build result UI
        preview.innerHTML = `
            <div class="flex flex-col md:flex-row items-center justify-center gap-8 md:gap-16 w-full max-w-[1000px] mt-12 mx-auto" id="resultBox">
                <div class="flex flex-col items-center text-center gap-6 max-w-[450px]">
                    <div class="relative w-[170px] h-[170px] flex items-center justify-center">
                        <svg class="absolute w-full h-full -rotate-90">
                            <circle cx="50%" cy="50%" r="78" stroke="#ddd" stroke-width="15" fill="none"/>
                            <circle id="progressCircle" cx="50%" cy="50%" r="78" stroke="#e26215" stroke-width="15" fill="none" stroke-linecap="round" stroke-dasharray="490" stroke-dashoffset="490"/>
                        </svg>
                        <span id="progressText" class="text-4xl font-bold text-[#7a3f1a] leading-none">0%</span>
                    </div>
                    <div class="flex flex-col items-center gap-3">
                        <h3 id="breedName" class="font-poppins font-extrabold text-[26px] text-[#e26215]">${top.breed.replace(/_/g," ").toUpperCase()}</h3>
                        <p id="breedDesc" class="italic text-[#555] text-[16px] leading-[1.6] text-center max-w-[420px]">Prediction confidence: ${confidence}%. This is the model's best guess based on visual features.</p>
                    </div>
                </div>
                <div class="flex flex-col items-center gap-5">
                    <img id="breedImage" src="${exampleImg}" alt="Example" class="w-[350px] h-[220px] object-cover rounded-xl shadow-md border-2 border-[#e26215]" />
                    <div class="flex gap-3">
                        <a id="downloadReport" href="#" class="btn-ghost font-poppins font-bold text-[18px] px-8 py-3 rounded-full border-4 border-[#e26215] bg-transparent text-[#e26215] no-underline inline-block transition-all duration-300 hover:-translate-y-1 hover:bg-[#e26215] hover:text-white hover:border-[#cc500f]">PRINT REPORT</a>
                    </div>
                </div>
            </div>
        `;
        animateElement(document.getElementById("resultBox"), {opacity:[0,1], translateY:[40,0], duration:700});
        animateElement(document.getElementById("breedImage"), {opacity:[0,1], scale:[0.9,1], duration:800, delay:200});
        animateProgress(confidence, document.getElementById("progressCircle"), document.getElementById("progressText"),1500);

    } catch(err){
        console.error("Prediction failed:", err);
        preview.innerHTML = `<p class="text-red-600">Error: ${err.message}</p>`;
    }
});

// -------------------------
// PDF generation
// -------------------------
document.addEventListener("click", async (e)=>{
    if(e.target && e.target.id==="downloadReport"){
        e.preventDefault();
        if(!lastPrediction || !uploadedFile){
            alert("Please upload an image first!");
            return;
        }

        const downloadBtn = e.target;
        downloadBtn.disabled = true;
        downloadBtn.textContent = "GENERATING...";

        const formData = new FormData();
        formData.append("breed", lastPrediction.breed);
        formData.append("confidence", lastPrediction.confidence);
        formData.append("image", uploadedFile, uploadedFile.name); // explicitly include filename

        try{
            const res = await fetch(GENERATE_PDF_URL, { method:"POST", body:formData });
            if(!res.ok) throw new Error(`Server returned ${res.status}`);
            const data = await res.json();

            if(data.pdf_url){
            const link = document.createElement("a");
            link.href = data.pdf_url;
            link.download = `${lastPrediction.breed}_report.pdf`;
            document.body.appendChild(link);
            link.click();
            link.remove();
        }

        } catch(err){
            console.error("PDF generation failed:", err);
            alert(`Error generating PDF report: ${err.message || "Server error"}`);
        } finally {
            downloadBtn.disabled = false;
            downloadBtn.textContent = "PRINT REPORT";
        }
    }
});
