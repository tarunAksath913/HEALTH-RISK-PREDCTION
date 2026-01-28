import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";

// --- Reusable Button ---
const Button = ({ children, variant = "primary", ...props }) => {
  const base = "inline-flex items-center justify-center rounded-md px-6 py-3 text-sm font-medium transition focus:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 lg:text-base";
  const variants = {
    primary: "bg-teal-600 text-white hover:bg-teal-700 focus-visible:ring-teal-600",
    outline: "border border-teal-600 text-teal-700 hover:bg-teal-50 focus-visible:ring-teal-600",
    link: "text-teal-700 hover:underline px-0 py-0",
  };
  return <button className={`${base} ${variants[variant]}`} {...props}>{children}</button>;
};

// --- Form Components ---
const Input = ({ label, name, type = "text", placeholder, value, onChange, min, max }) => (
  <div className="block">
    <label className="text-xs font-medium text-gray-700">{label}</label>
    <input name={name} type={type} placeholder={placeholder} value={value} onChange={onChange} min={min} max={max} className="mt-1 w-full rounded-md border border-gray-300 px-3 py-2 text-sm focus:border-teal-600 focus:outline-none focus:ring-1 focus:ring-teal-600 placeholder:text-gray-400" />
  </div>
);

const Select = ({ label, name, value, onChange, options }) => (
  <div className="block">
    <label className="text-xs font-medium text-gray-700">{label}</label>
    <select name={name} value={value} onChange={onChange} className="mt-1 w-full rounded-md border border-gray-300 px-3 py-2 text-sm text-gray-700 focus:border-teal-600 focus:outline-none focus:ring-1 focus:ring-teal-600">
      {options.map((opt) => (
        <option key={opt} value={opt}>{opt}</option>
      ))}
    </select>
  </div>
);

// --- Steps ---
const StepOne = ({ data, onChange }) => (
  <div className="space-y-4">
    <Input label="Full Name" name="fullName" value={data.fullName} onChange={onChange} placeholder="John Doe" />
    <Input label="Age" name="age" type="number" min="18" max="100" value={data.age} onChange={onChange} placeholder="25" />
    <Select label="Gender" name="gender" value={data.gender} onChange={onChange} options={["Male", "Female"]} />
    <Select label="Smoking History" name="smoking_history" value={data.smoking_history} onChange={onChange} options={["never", "current", "former", "No Info"]} />
  </div>
);

const StepTwo = ({ data, onChange }) => (
  <div className="space-y-4">
    <Input label="Weight (kg)" name="weight" type="number" value={data.weight} onChange={onChange} placeholder="70" />
    <Input label="Height (m)" name="height" type="number" step="0.01" value={data.height} onChange={onChange} placeholder="1.75" />
    <Input label="BMI (Auto-calculated)" name="bmi" value={data.bmi} readOnly placeholder="22.5" />
  </div>
);

const StepThree = ({ data, onChange }) => (
  <div className="space-y-4">
    <Select label="How often do you snack?" name="snackFreq" value={data.snackFreq} onChange={onChange} options={["no", "Sometimes", "Frequently", "Always"]} />
    <Select label="Alcohol Consumption" name="alcohol" value={data.alcohol} onChange={onChange} options={["no", "Sometimes", "Frequently", "Always"]} />
    <Select label="Transportation Mode" name="transport" value={data.transport} onChange={onChange} options={["Public_Transportation", "Automobile", "Walking", "Motorbike", "Bike"]} />
    <label className="flex items-center gap-2">
      <input type="checkbox" name="highCalFood" checked={data.highCalFood} onChange={onChange} className="accent-teal-600" />
      <span className="text-sm">Frequent High Calorie Food?</span>
    </label>
  </div>
);

const StepFour = ({ data, onChange }) => (
  <div className="space-y-4">
    <p className="text-gray-500 text-xs mb-2">Medical History (Check if Yes)</p>
    <label className="flex items-center gap-2"><input type="checkbox" name="hypertension" checked={data.hypertension} onChange={onChange} className="accent-teal-600" /> Hypertension</label>
    <label className="flex items-center gap-2"><input type="checkbox" name="heart_disease" checked={data.heart_disease} onChange={onChange} className="accent-teal-600" /> Heart Disease</label>
    <label className="flex items-center gap-2"><input type="checkbox" name="historyObesity" checked={data.historyObesity} onChange={onChange} className="accent-teal-600" /> Family History of Obesity</label>
  </div>
);

// --- Main Component ---
export default function CreateProfile() {
  const [step, setStep] = useState(1);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();
  
  const [formData, setFormData] = useState({
    fullName: "", age: "25", gender: "Male", smoking_history: "never",
    weight: "70", height: "1.75", bmi: "22.86",
    snackFreq: "Sometimes", alcohol: "Sometimes", transport: "Public_Transportation",
    highCalFood: true, hypertension: false, heart_disease: false, historyObesity: true,
    // Defaults for hidden fields
    veggieFreq: 2.0, mealsPerDay: 3.0, waterIntake: 2.0, activityFreq: 1.0, techUse: 1.0
  });

  // Calculate BMI automatically when weight/height change
  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    const newData = { ...formData, [name]: type === "checkbox" ? checked : value };
    
    if (name === "weight" || name === "height") {
      const w = parseFloat(name === "weight" ? value : formData.weight);
      const h = parseFloat(name === "height" ? value : formData.height);
      if (w > 0 && h > 0) {
        newData.bmi = (w / (h * h)).toFixed(2);
      }
    }
    setFormData(newData);
  };

  const handleSubmit = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:5000/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData),
      });
      const result = await response.json();
      console.log("AI Prediction:", result);
      
      // Navigate to dashboard with the result
      navigate("/dashboard", { state: { prediction: result } });
      
    } catch (error) {
      console.error("Error submitting form:", error);
      alert("Failed to connect to AI server.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="flex min-h-screen items-center justify-center bg-gray-50 px-4 font-sans">
      <div className="w-full max-w-md bg-white p-6 rounded-lg shadow-sm">
        <h1 className="text-2xl font-semibold text-center text-gray-900 mb-6">Create Your Profile {step}/4</h1>
        
        {step === 1 && <StepOne data={formData} onChange={handleChange} />}
        {step === 2 && <StepTwo data={formData} onChange={handleChange} />}
        {step === 3 && <StepThree data={formData} onChange={handleChange} />}
        {step === 4 && <StepFour data={formData} onChange={handleChange} />}

        <div className="mt-6 flex justify-end gap-3">
          {step > 1 && <Button variant="outline" onClick={() => setStep(step - 1)}>Back</Button>}
          
          {step < 4 ? (
            <Button onClick={() => setStep(step + 1)}>Continue</Button>
          ) : (
            <Button onClick={handleSubmit} disabled={loading}>
              {loading ? "Analyzing..." : "Get Prediction"}
            </Button>
          )}
        </div>
      </div>
    </main>
  );
}