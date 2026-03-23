async function predict() {
  const payload = {
    Gender:            +document.getElementById('Gender').value,
    Married:           +document.getElementById('Married').value,
    Dependents:        +document.getElementById('Dependents').value,
    Education:         +document.getElementById('Education').value,
    Self_Employed:     +document.getElementById('Self_Employed').value,
    ApplicantIncome:   +document.getElementById('ApplicantIncome').value,
    CoapplicantIncome: +document.getElementById('CoapplicantIncome').value,
    LoanAmount:        +document.getElementById('LoanAmount').value,
    Loan_Amount_Term:  +document.getElementById('Loan_Amount_Term').value,
    Credit_History:    +document.getElementById('Credit_History').value,
    Property_Area:     +document.getElementById('Property_Area').value,
  };

  document.getElementById('loader').style.display    = 'block';
  document.getElementById('result-box').style.display = 'none';
  document.getElementById('explanation').style.display = 'none';

  try {
    const res  = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();

    document.getElementById('loader').style.display = 'none';

    // ── Result ──────────────────────────────────────────────
    const box = document.getElementById('result-box');
    box.style.display = 'block';
    box.className = data.prediction.includes('APPROVED') ? 'approved' : 'rejected';
    document.getElementById('result-label').textContent = data.prediction;
    document.getElementById('result-prob').textContent  = `Confidence: ${data.probability}%`;

    // ── SHAP Bars ────────────────────────────────────────────
    const container = document.getElementById('shap-bars');
    container.innerHTML = '';
    const sorted = Object.entries(data.shap_values)
      .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 8);
    const maxVal = Math.max(...sorted.map(([, v]) => Math.abs(v)));

    sorted.forEach(([feat, val]) => {
      const pct   = (Math.abs(val) / maxVal * 100).toFixed(1);
      const color = val >= 0 ? '#4ade80' : '#f87171';
      container.innerHTML += `
        <div class="shap-row">
          <div class="shap-label">${feat}</div>
          <div class="bar-wrap">
            <div class="bar-fill" style="width:${pct}%;background:${color}"></div>
          </div>
          <div class="shap-num" style="color:${color}">
            ${val > 0 ? '+' : ''}${val.toFixed(3)}
          </div>
        </div>`;
    });

    // ── SHAP Image ───────────────────────────────────────────
    document.getElementById('shap-img').src =
      'data:image/png;base64,' + data.shap_plot;

    // ── LIME Rules ───────────────────────────────────────────
    const limeList = document.getElementById('lime-list');
    limeList.innerHTML = '';
    data.lime_explanation.forEach(line => {
      limeList.innerHTML += `<li>${line}</li>`;
    });
    

    document.getElementById('explanation').style.display = 'block';

  } catch (err) {
    document.getElementById('loader').style.display = 'none';
    alert('Error connecting to server: ' + err.message);
  }
}