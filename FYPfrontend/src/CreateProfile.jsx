import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";

// --- Reusable UI Components ---
const Button = ({ children, variant = "primary", ...props }) => {
  const base = "inline-flex items-center justify-center rounded-md px-6 py-3 text-sm font-medium transition focus:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 lg:text-base";
  const variants = {
    primary: "bg-teal-600 text-white hover:bg-teal-700 focus-visible:ring-teal-600",
    outline: "border border-teal-600 text-teal-700 hover:bg-teal-50 focus-visible:ring-teal-600",
    link: "text-teal-700 hover:underline px-0 py-0",
  };
  return <button className={`${base} ${variants[variant]}`} {...props}>{children}</button>;
};

const Input = ({ label, name, type = "text", placeholder, value, onChange, min, max, step }) => (
  <div className="block">
    <label className="text-xs font-medium text-gray-700">{label}</label>
    <input 
      name={name} type={type} placeholder={placeholder} 
      value={value} onChange={onChange} min={min} max={max} step={step}
      className="mt-1 w-full rounded-md border border-gray-300 px-3 py-2 text-sm focus:border-teal-600 focus:outline-none focus:ring-1 focus:ring-teal-600 placeholder:text-gray-400" 
    />
  </div>
);

const Select = ({ label, name, value, onChange, options }) => (
  <div className="block">
    <label className="text-xs font-medium text-gray-700">{label}</label>
    <select 
      name={name} value={value} onChange={onChange} 
      className="mt-1 w-full rounded-md border border-gray-300 px-3 py-2 text-sm text-gray-700 focus:border-teal-600 focus:outline-none focus:ring-1 focus:ring-teal-600"
    >
      {options.map((opt) => (
        <option key={opt.value || opt} value={opt.value || opt}>
          {opt.label || opt}
        </option>
      ))}
    </select>
  </div>
);

const Checkbox = ({ label, name, checked, onChange }) => (
  <label className="flex items-center gap-3 p-3 border rounded-md hover:bg-gray-50 cursor-pointer transition">
    <input 
      type="checkbox" name={name} checked={checked} onChange={onChange} 
      className="h-4 w-4 accent-teal-600 rounded border-gray-300 focus:ring-teal-500" 
    />
    <span className="text-sm font-medium text-gray-700">{label}</span>
  </label>
);

// --- Step 1: Demographics ---
const StepOne = ({ data, onChange }) => (
  <div className="space-y-4 animate-in fade-in slide-in-from-right-4 duration-300">
    <h3 className="text-lg font-medium text-teal-800 border-b pb-2">Basic Information</h3>
    <div className="space-y-4">
      <Input label="Full Name" name="fullName" value={data.fullName} onChange={onChange} placeholder="John Doe" />
      <Input label="Email Address" name="email" type="email" value={data.email} onChange={onChange} placeholder="john@example.com" />
    </div>
    <div className="grid grid-cols-2 gap-4">
      <Input label="Age" name="age" type="number" min="1" max="120" value={data.age} onChange={onChange} placeholder="e.g. 25" />
      <Select label="Gender" name="gender" value={data.gender} onChange={onChange} options={["Male", "Female"]} />
    </div>
    <div className="grid grid-cols-2 gap-4">
      <Input label="Weight (kg)" name="weight" type="number" value={data.weight} onChange={onChange} placeholder="e.g. 70" />
      <Input label="Height (m)" name="height" type="number" step="0.01" value={data.height} onChange={onChange} placeholder="e.g. 1.75" />
    </div>
    <Input label="BMI (Auto-calculated)" name="bmi" value={data.bmi} readOnly placeholder="--" />
    <Checkbox label="Family History of Overweight?" name="family_history" checked={data.family_history} onChange={onChange} />
  </div>
);

// --- Step 2: Medical History ---
const StepTwo = ({ data, onChange }) => (
  <div className="space-y-4 animate-in fade-in slide-in-from-right-4 duration-300">
    <h3 className="text-lg font-medium text-teal-800 border-b pb-2">Medical History</h3>
    <p className="text-xs text-gray-500">Have you ever been told by a doctor that you have...</p>
    <div className="space-y-2">
      <Checkbox label="High Blood Pressure (Hypertension)" name="highBP" checked={data.highBP} onChange={onChange} />
      <Checkbox label="High Cholesterol" name="highChol" checked={data.highChol} onChange={onChange} />
      <Checkbox label="Stroke" name="stroke" checked={data.stroke} onChange={onChange} />
      <Checkbox label="Heart Disease or Heart Attack" name="heartDisease" checked={data.heartDisease} onChange={onChange} />
      <Checkbox label="Difficulty Walking or Climbing Stairs" name="diffWalk" checked={data.diffWalk} onChange={onChange} />
    </div>
  </div>
);

// --- Step 3: Lifestyle & Habits ---
const StepThree = ({ data, onChange }) => (
  <div className="space-y-4 animate-in fade-in slide-in-from-right-4 duration-300">
    <h3 className="text-lg font-medium text-teal-800 border-b pb-2">Lifestyle & Habits</h3>
    <div className="grid grid-cols-2 gap-4">
      <Select label="Do you smoke?" name="smoker" value={data.smoker} onChange={onChange} options={[{label: "No", value: false}, {label: "Yes", value: true}]} />
      <Select label="Monitor Calories?" name="monitorCalories" value={data.monitorCalories} onChange={onChange} options={[{label: "No", value: false}, {label: "Yes", value: true}]} />
    </div>
    <Select label="Alcohol Consumption" name="alcohol" value={data.alcohol} onChange={onChange} options={["no", "Sometimes", "Frequently", "Always"]} />
    <Input label="Physical Activity (Days per week)" name="physActivityDays" type="number" min="0" max="7" value={data.physActivityDays} onChange={onChange} />
    <Input label="Tech Use (Hours per day)" name="techUseHours" type="number" min="0" max="24" value={data.techUseHours} onChange={onChange} />
    <Select label="Primary Transport" name="transport" value={data.transport} onChange={onChange} options={["Public_Transportation", "Automobile", "Walking", "Motorbike", "Bike"]} />
  </div>
);

// --- Step 4: Diet & Nutrition ---
const StepFour = ({ data, onChange }) => (
  <div className="space-y-4 animate-in fade-in slide-in-from-right-4 duration-300">
    <h3 className="text-lg font-medium text-teal-800 border-b pb-2">Diet & Nutrition</h3>
    
    <div className="grid grid-cols-2 gap-4">
      <Select label="Veggie Frequency" name="veggieFreq" value={data.veggieFreq} onChange={onChange} options={["Never", "Sometimes", "Always"]} />
      <Select label="Daily Fruit?" name="fruitDaily" value={data.fruitDaily} onChange={onChange} options={[{label: "No", value: false}, {label: "Yes", value: true}]} />
    </div>

    <div className="grid grid-cols-2 gap-4">
      <Input label="Meals per day" name="mealsPerDay" type="number" min="1" max="10" value={data.mealsPerDay} onChange={onChange} />
      <Select label="Water Intake" name="waterDaily" value={data.waterDaily} onChange={onChange} options={[{label: "< 1 Liter", value: 1}, {label: "1-2 Liters", value: 2}, {label: "> 2 Liters", value: 3}]} />
    </div>

    <Select label="Snacking Frequency" name="snackFreq" value={data.snackFreq} onChange={onChange} options={["no", "Sometimes", "Frequently", "Always"]} />
    <Checkbox label="Frequent High Calorie Food?" name="highCalFood" checked={data.highCalFood} onChange={onChange} />
  </div>
);

// --- Step 5: Recent Health ---
const StepFive = ({ data, onChange }) => (
  <div className="space-y-4 animate-in fade-in slide-in-from-right-4 duration-300">
    <h3 className="text-lg font-medium text-teal-800 border-b pb-2">Recent Health (Last 30 Days)</h3>
    
    <div className="block">
      <label className="text-xs font-medium text-gray-700">General Health Rating (1=Excellent, 5=Poor)</label>
      <input 
        type="range" name="genHealth" min="1" max="5" value={data.genHealth} onChange={onChange} 
        className="w-full mt-2 accent-teal-600"
      />
      <div className="flex justify-between text-xs text-gray-500 px-1">
        <span>Excellent (1)</span>
        <span>Poor (5)</span>
      </div>
      <div className="text-center font-bold text-teal-700 mt-1">Selected: {data.genHealth}</div>
    </div>

    <Input label="Days with POOR Physical Health (Illness/Injury)" name="physHealthDays" type="number" min="0" max="30" value={data.physHealthDays} onChange={onChange} />
    <Input label="Days with POOR Mental Health (Stress/Depression)" name="mentHealthDays" type="number" min="0" max="30" value={data.mentHealthDays} onChange={onChange} />
  </div>
);

// --- Main Component ---
export default function CreateProfile() {
  const [step, setStep] = useState(1);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();
  
  // Initialize state with all 25 fields
  const [formData, setFormData] = useState({
    // Step 1: Demographics
    fullName: "", email: "", age: "", gender: "Male", height: "", weight: "", bmi: "", family_history: false,
    
    // Step 2: Medical
    highBP: false, highChol: false, stroke: false, heartDisease: false, diffWalk: false,
    
    // Step 3: Lifestyle
    smoker: false, alcohol: "Sometimes", physActivityDays: 0, techUseHours: 0, transport: "Public_Transportation", monitorCalories: false,
    
    // Step 4: Diet
    veggieFreq: "Sometimes", fruitDaily: false, highCalFood: false, mealsPerDay: 3, snackFreq: "Sometimes", waterDaily: 2,
    
    // Step 5: Recent Health
    genHealth: 3, physHealthDays: 0, mentHealthDays: 0
  });

  // --- NEW: Validation Helper ---
  // Returns true only if essential fields have real numbers
  const isFormValid = () => {
    return (
      formData.fullName !== "" &&
      formData.email !== "" &&
      formData.age !== "" &&
      formData.height !== "" &&
      formData.weight !== "" &&
      formData.bmi !== "" // Ensures height/weight were valid enough to calculate BMI
    );
  };

  // Auto-Calculate BMI
  useEffect(() => {
    const w = parseFloat(formData.weight);
    const h = parseFloat(formData.height);
    if (w > 0 && h > 0) {
      setFormData(prev => ({ ...prev, bmi: (w / (h * h)).toFixed(2) }));
    }
  }, [formData.weight, formData.height]);

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    // For selects that use boolean values (like smoker), we need to handle "true"/"false" strings
    let finalValue = type === "checkbox" ? checked : value;
    
    if (value === "true") finalValue = true;
    if (value === "false") finalValue = false;

    setFormData(prev => ({ ...prev, [name]: finalValue }));
  };

const handleSubmit = async () => {
    setLoading(true);
    try {
      // 1. Call your Node.js Backend
      const response = await fetch('http://localhost:5000/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData),
      });

      if (!response.ok) throw new Error("Network response was not ok");
      
      const result = await response.json();
      console.log("AI Prediction:", result);
      
      // 2. Navigate to Dashboard with BOTH Prediction and User Data
      navigate("/dashboard", { 
        state: { 
          prediction: result,      // The AI result
          profileData: formData    // The User's input (weight, etc.)
        } 
      });
      
    } catch (error) {
      console.error("Error submitting form:", error);
      alert("Failed to connect to AI server. Check console.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="flex min-h-screen items-center justify-center bg-gray-50 px-4 font-sans py-8">
      <div className="w-full max-w-lg bg-white p-8 rounded-xl shadow-lg border border-gray-100">
        
        {/* ... (keep Progress Bar the same) ... */}
        
        {/* Form Steps */}
        <div className="min-h-[320px]">
          {step === 1 && <StepOne data={formData} onChange={handleChange} />}
          {step === 2 && <StepTwo data={formData} onChange={handleChange} />}
          {step === 3 && <StepThree data={formData} onChange={handleChange} />}
          {step === 4 && <StepFour data={formData} onChange={handleChange} />}
          {step === 5 && <StepFive data={formData} onChange={handleChange} />}
        </div>

        {/* Navigation Buttons */}
        <div className="mt-8 flex justify-between pt-4 border-t border-gray-100">
          {step > 1 ? (
            <Button variant="outline" onClick={() => setStep(step - 1)}>Back</Button>
          ) : <div />}
          
          {step < 5 ? (
            // Only allow 'Continue' on Step 1 if basic data is filled
            <Button 
              onClick={() => setStep(step + 1)}
              // Optional: You can disable "Continue" on Step 1 specifically
              disabled={step === 1 && !isFormValid()} 
              className={step === 1 && !isFormValid() ? "opacity-50 cursor-not-allowed" : ""}
            >
              Continue
            </Button>
          ) : (
            // Disable Submit if form is invalid
            <Button 
              onClick={handleSubmit} 
              disabled={loading || !isFormValid()} 
              className={`bg-teal-800 hover:bg-teal-900 ${(!isFormValid() || loading) ? "opacity-50 cursor-not-allowed" : ""}`}
            >
              {loading ? "Analyzing..." : "Get Prediction"}
            </Button>
          )}
        </div>
      </div>
    </main>
  );
}