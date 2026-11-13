const get = (id) => document.getElementById(id);
const profile = get("profile");
const scoreButton = get("score");
const exampleButton = get("example");
const groups = [
  ["Plan and billing", {
    Contract: "Contract", tenure: "Tenure (months)", MonthlyCharges: "Monthly charges",
    TotalCharges: "Total charges", PaymentMethod: "Payment method", PaperlessBilling: "Paperless billing",
  }],
  ["Services", {
    InternetService: "Internet service", PhoneService: "Phone service", MultipleLines: "Multiple lines",
    OnlineSecurity: "Online security", OnlineBackup: "Online backup", DeviceProtection: "Device protection",
    TechSupport: "Tech support", StreamingTV: "Streaming TV", StreamingMovies: "Streaming movies",
  }],
  ["Household", { gender: "Gender", SeniorCitizen: "Senior citizen", Partner: "Partner", Dependents: "Dependents" }],
];

async function check(response) {
  if (!response.ok) {
    const payload = await response.json();
    throw new Error(payload.error || "The request could not be completed.");
  }
  return response;
}

function renderFields(settings) {
  for (const [groupName, fields] of groups) {
    const fieldset = document.createElement("fieldset");
    const legend = document.createElement("legend");
    legend.textContent = groupName;
    const grid = document.createElement("div");
    grid.className = "inputs";
    for (const [name, label] of Object.entries(fields)) {
      const wrapper = document.createElement("div");
      const caption = document.createElement("label");
      caption.htmlFor = name;
      caption.textContent = label;
      const categories = settings.categories[name];
      const control = document.createElement(categories ? "select" : "input");
      control.id = name;
      control.name = name;
      control.required = true;
      if (categories) {
        for (const value of categories) {
          const display = name === "SeniorCitizen" ? (value === "1" ? "Yes" : "No") : value;
          control.add(new Option(display, value));
        }
      } else {
        control.type = "number";
        control.min = "0";
        control.step = name === "tenure" ? "1" : "0.01";
      }
      wrapper.append(caption, control);
      grid.append(wrapper);
    }
    fieldset.append(legend, grid);
    get("fields").append(fieldset);
  }
}

function clearResult(message) {
  get("details").hidden = true;
  get("placeholder").hidden = false;
  get("placeholder").textContent = message;
  get("profile-error").textContent = "";
}

async function initialize() {
  try {
    const response = await check(await fetch("/api/settings"));
    const settings = await response.json();
    renderFields(settings);
    const loadExample = () => {
      for (const [name, value] of Object.entries(settings.example)) {
        profile.elements.namedItem(name).value = value;
      }
      clearResult("Submit a profile to see its score and risk band.");
    };
    exampleButton.addEventListener("click", loadExample);
    loadExample();
    exampleButton.disabled = false;
    scoreButton.disabled = false;
  } catch (error) {
    get("profile-error").textContent = error.message;
  }
}

profile.addEventListener("input", () => clearResult("Submit the updated profile to refresh its score."));
profile.addEventListener("submit", async (event) => {
  event.preventDefault();
  scoreButton.disabled = true;
  exampleButton.disabled = true;
  const profileData = Object.fromEntries(new FormData(profile));
  get("fields").inert = true;
  clearResult("Scoring profile…");
  try {
    const response = await check(await fetch("/api/score", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(profileData),
    }));
    const result = await response.json();
    get("value").textContent = result.risk_score.toFixed(3);
    get("band").textContent = result.risk_segment;
    get("band").dataset.band = result.risk_segment;
    get("meter").value = result.risk_score;
    get("flag").textContent = result.flag_at_0_40 ? "Flagged at the 0.40 threshold" : "Below the 0.40 flag threshold";
    get("placeholder").hidden = true;
    get("details").hidden = false;
  } catch (error) {
    clearResult("The profile could not be scored.");
    get("profile-error").textContent = error.message;
  } finally {
    get("fields").inert = false;
    scoreButton.disabled = false;
    exampleButton.disabled = false;
  }
});

get("batch").addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = get("file").files[0];
  const button = get("batch-submit");
  const status = get("batch-status");
  status.classList.remove("error");
  status.textContent = "Scoring CSV…";
  button.disabled = true;
  try {
    const response = await check(await fetch("/api/batch", {
      method: "POST", headers: { "Content-Type": "text/csv" }, body: file,
    }));
    const url = URL.createObjectURL(await response.blob());
    const link = document.createElement("a");
    link.href = url;
    link.download = "customer-risk-scores.csv";
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    status.textContent = "Scored CSV downloaded.";
  } catch (error) {
    status.classList.add("error");
    status.textContent = error.message;
  } finally {
    button.disabled = false;
  }
});

initialize();
