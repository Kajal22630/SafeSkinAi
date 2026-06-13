document.addEventListener("DOMContentLoaded", function () {

  const API_BASE = "http://127.0.0.1:8000";

  const form = document.getElementById("predictForm");
  const resultContainer = document.getElementById("resultContainer");

  const diagnosisText = document.getElementById("diagnosisText");
  const confidenceText = document.getElementById("confidenceText");
  const riskBar = document.getElementById("riskBar");
  const riskLevel = document.getElementById("riskLevel");
  const reportIDText = document.getElementById("reportID");

  const originalImage = document.getElementById("originalImage");
  const gradcamImage = document.getElementById("gradcamImage");

  const compareSlider = document.getElementById("compareSlider");
  const overlayContainer = document.getElementById("overlayContainer");
  const sliderLine = document.getElementById("sliderLine");

  const intensitySlider = document.getElementById("intensitySlider");

  const tableBody = document.getElementById("historyTableBody");

  /* =====================================================
     FORM SUBMIT
  ===================================================== */

  if (form) {

    form.addEventListener("submit", async function (e) {

      e.preventDefault();

      const name = document.getElementById("patient_name").value.trim();
      const age = parseInt(document.getElementById("age").value);
      const gender = document.getElementById("gender").value;
      const doctor = document.getElementById("doctor_name").value.trim();
      const file = document.getElementById("image").files[0];

      if (!name || !doctor) return alert("Enter valid names");
      if (!age || age < 1 || age > 120) return alert("Invalid age");
      if (!gender) return alert("Select gender");
      if (!file) return alert("Upload image");

      const submitBtn = form.querySelector("button[type='submit']");

      submitBtn.disabled = true;
      submitBtn.innerHTML =
        `<span class="spinner-border spinner-border-sm"></span> Processing...`;

      const formData = new FormData();

      formData.append("patient_name", name);
      formData.append("age", age);
      formData.append("gender", gender);
      formData.append("doctor_name", doctor);
      formData.append("file", file);

      try {

        const controller = new AbortController();

        const timeout = setTimeout(() => controller.abort(), 60000);

        const response = await fetch(`${API_BASE}/predict`, {
          method: "POST",
          body: formData,
          signal: controller.signal
        });

        clearTimeout(timeout);

        if (!response.ok) {

          const text = await response.text();

          throw new Error(text || "Prediction failed");
        }

        const data = await response.json();

        if (!data.report_id) {
          throw new Error("Invalid server response");
        }

        localStorage.setItem("lastResult", JSON.stringify(data));

        displayResult(data);

        if (tableBody) loadHistory();

        form.reset();

      } catch (err) {

        if (err.name === "AbortError") {
          alert("Server timeout. Please try again.");
        } else {
          alert("Error: " + err.message);
        }

        console.error(err);

      } finally {

        submitBtn.disabled = false;

        submitBtn.innerHTML =
          `<i class="bi bi-cpu"></i> Analyze Skin Lesion`;
      }

    });

  }


  /* =====================================================
     RESTORE LAST RESULT
  ===================================================== */

  const savedResult = localStorage.getItem("lastResult");

  if (savedResult && resultContainer) {

    try {

      const data = JSON.parse(savedResult);

      fetch(`${API_BASE}/generated_reports/${data.report_id}_original.jpg`)
        .then(res => {

          if (res.ok) {
            displayResult(data);
          } else {
            localStorage.removeItem("lastResult");
          }

        });

    } catch {

      localStorage.removeItem("lastResult");

    }

  }


  /* =====================================================
     DISPLAY RESULT
  ===================================================== */

  function displayResult(data) {

    if (!resultContainer) return;

    resultContainer.classList.remove("d-none");

    reportIDText.textContent = data.report_id;

    diagnosisText.textContent = data.diagnosis;

    riskLevel.textContent = data.risk;

    diagnosisText.className =
      data.diagnosis === "Malignant"
        ? "fw-bold text-danger"
        : "fw-bold text-success";

    const prob = parseFloat(data.probability);

    animateCounter(prob);

    riskBar.style.width = prob + "%";

    riskBar.textContent = prob.toFixed(1) + "%";

    loadImages(data.report_id);

    resultContainer.scrollIntoView({ behavior: "smooth" });

  }


  /* =====================================================
     IMAGE LOADING
  ===================================================== */

  function loadImages(reportID) {

    const timestamp = Date.now();

    const originalURL =
      `${API_BASE}/generated_reports/${reportID}_original.jpg?t=${timestamp}`;

    const gradcamURL =
      `${API_BASE}/generated_reports/${reportID}_gradcam.jpg?t=${timestamp}`;

    if (originalImage) originalImage.src = originalURL;

    if (gradcamImage) gradcamImage.src = gradcamURL;

    if (originalImage)
      originalImage.onerror = () =>
        retryImage(originalImage, originalURL);

    if (gradcamImage)
      gradcamImage.onerror = () =>
        retryImage(gradcamImage, gradcamURL);

  }


  function retryImage(img, url, retries = 5) {

    if (retries <= 0) return;

    setTimeout(() => {

      img.src = url + "&retry=" + retries;

      retryImage(img, url, retries - 1);

    }, 500);

  }


  /* =====================================================
     CONFIDENCE ANIMATION
  ===================================================== */

  function animateCounter(target) {

    let current = 0;

    const step = target / 40;

    const interval = setInterval(() => {

      current += step;

      if (current >= target) {

        current = target;

        clearInterval(interval);

      }

      confidenceText.textContent =
        current.toFixed(1) + "%";

    }, 25);

  }


  /* =====================================================
     IMAGE COMPARISON SLIDER
  ===================================================== */

  if (compareSlider && overlayContainer && sliderLine) {

    const updateSlider = (value) => {

      overlayContainer.style.width = value + "%";

      sliderLine.style.left = value + "%";

    };

    updateSlider(compareSlider.value);

    compareSlider.addEventListener("input", function () {

      updateSlider(this.value);

    });

  }


  /* =====================================================
     HEATMAP INTENSITY CONTROL
  ===================================================== */

  if (intensitySlider && gradcamImage) {

    intensitySlider.addEventListener("input", function () {

      gradcamImage.style.opacity = this.value / 100;

    });

  }


  /* =====================================================
     LOAD HISTORY
  ===================================================== */

  if (tableBody) loadHistory();

  async function loadHistory() {

    try {

      tableBody.innerHTML =
        `<tr><td colspan="7" class="text-center">Loading reports...</td></tr>`;

      const res = await fetch(`${API_BASE}/history`);

      const data = await res.json();

      tableBody.innerHTML = "";

      if (!data.length) {

        tableBody.innerHTML =
          `<tr><td colspan="7" class="text-center">No Reports Found</td></tr>`;

        return;

      }

      data.sort((a, b) =>
        b.report_id.localeCompare(a.report_id)
      );

      data.forEach(report => {

        const badge =
          report.diagnosis === "Malignant"
            ? `<span class="badge bg-danger">Malignant</span>`
            : `<span class="badge bg-success">Benign</span>`;

        tableBody.insertAdjacentHTML(
          "beforeend",
          `
          <tr>
            <td>${report.report_id}</td>
            <td>${report.patient_name}</td>
            <td>${report.age}</td>
            <td>${report.doctor_name}</td>
            <td>${badge}</td>
            <td>${report.probability}%</td>
            <td>

              <a href="${API_BASE}/generate-report/${report.report_id}"
                 target="_blank"
                 class="btn btn-sm btn-success">
                 <i class="bi bi-file-earmark-pdf"></i>
              </a>

              <button class="btn btn-sm btn-danger delete-btn"
                      data-id="${report.report_id}">
                <i class="bi bi-trash"></i>
              </button>

            </td>
          </tr>
          `
        );

      });

    } catch (err) {

      console.error("History load error:", err);

    }

  }


  /* =====================================================
     DELETE REPORT
  ===================================================== */

  document.addEventListener("click", async function (e) {

    const btn = e.target.closest(".delete-btn");

    if (!btn) return;

    if (!confirm("Delete this report?")) return;

    const id = btn.dataset.id;

    try {

      const res = await fetch(`${API_BASE}/delete-report/${id}`, {
        method: "DELETE"
      });

      if (res.ok) {

        btn.closest("tr").remove();

        if (!tableBody.children.length) {

          tableBody.innerHTML =
            `<tr><td colspan="7" class="text-center">No Reports Found</td></tr>`;

        }

      }

    } catch (err) {

      console.error("Delete error:", err);

    }

  });

});