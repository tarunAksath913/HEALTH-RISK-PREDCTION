import React, { useState } from "react";
import { BrowserRouter as Router, Routes, Route, Link, useNavigate } from "react-router-dom";
import CreateProfile from "./CreateProfile";
import Dashboard from "./Dashboard";
import MealPlan from "./MealPlan";
import ExercisePlan from "./ExercisePlan";
import Progress from "./Progress";



// --- Reusable Button ---
const Button = ({ children, variant = "primary", className = "", ...props }) => {
  const base = "inline-flex items-center justify-center rounded-md px-6 py-3 text-sm font-medium transition focus:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 lg:text-base";
  const variants = {
    primary: "bg-teal-600 text-white hover:bg-teal-700 focus-visible:ring-teal-600",
    outline: "border border-teal-600 text-teal-700 hover:bg-teal-50 focus-visible:ring-teal-600",
    link: "text-teal-700 hover:underline px-0 py-0",
  };

  return (
    <button className={`${base} ${variants[variant]} ${className}`} {...props}>
      {children}
    </button>
  );
};

// --- NEW COMPONENT: Login Modal ---
const LoginModal = ({ isOpen, onClose }) => {
  const navigate = useNavigate();
  
  if (!isOpen) return null;

  const handleLogin = (e) => {
    e.preventDefault();
    // Simulate checking "Existing User" -> Go to Dashboard
    navigate("/dashboard");
  };

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/50 backdrop-blur-sm p-4">
      <div className="w-full max-w-md rounded-xl bg-white p-8 shadow-2xl animate-in fade-in zoom-in duration-200">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-2xl font-bold text-gray-900">Welcome Back</h2>
          <button onClick={onClose} className="text-gray-400 hover:text-gray-600 text-2xl">&times;</button>
        </div>

        <form onSubmit={handleLogin} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Email</label>
            <input type="email" required className="w-full rounded-md border border-gray-300 px-4 py-2 focus:border-teal-500 focus:outline-none focus:ring-1 focus:ring-teal-500" placeholder="user@example.com" />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Password</label>
            <input type="password" required className="w-full rounded-md border border-gray-300 px-4 py-2 focus:border-teal-500 focus:outline-none focus:ring-1 focus:ring-teal-500" placeholder="••••••••" />
          </div>
          
          <Button className="w-full">Login</Button>
        </form>

        <div className="mt-6 text-center text-sm text-gray-600">
          Don't have an account?{" "}
          <Link to="/create-profile" className="font-semibold text-teal-600 hover:text-teal-500 hover:underline">
            Create Profile
          </Link>
        </div>
      </div>
    </div>
  );
};

// --- Header (Updated to accept onLoginClick) ---
const Header = ({ onLoginClick }) => (
  <header className="sticky top-0 z-50 bg-white shadow-sm">
    <nav className="mx-auto flex max-w-screen-2xl items-center justify-between px-6 py-4 md:px-10">
      <div className="text-lg font-semibold text-gray-900 lg:text-xl">
        Health Risk<br className="hidden sm:block" /> Prediction Website
      </div>
      <div className="hidden items-center gap-8 text-sm text-gray-700 font-medium md:flex lg:text-base">
        <a href="#about" className="hover:text-teal-700 transition">About</a>
        <a href="#features" className="hover:text-teal-700 transition">Features</a>
        <a href="#contact" className="hover:text-teal-700 transition">Contact</a>
        {/* Update: Button now triggers the modal */}
        <Button variant="outline" onClick={onLoginClick}>Login</Button>
      </div>
    </nav>
  </header>
);

// --- NEW COMPONENT: Learn More Modal ---
const LearnMoreModal = ({ isOpen, onClose }) => {
  if (!isOpen) return null;

  const steps = [
    {
      num: "01",
      title: "Create Profile",
      desc: "Enter your basic health stats (age, weight, height) and lifestyle habits in our secure form."
    },
    {
      num: "02",
      title: "AI Analysis",
      desc: "Our Machine Learning algorithm compares your data against clinical records to predict specific health risks."
    },
    {
      num: "03",
      title: "Get Your Plan",
      desc: "Receive a personalized meal and exercise plan scientifically designed to lower your specific risk factors."
    },
    {
      num: "04",
      title: "Track & Improve",
      desc: "Log your daily progress. As you build healthy habits, watch your risk score decrease in real-time."
    }
  ];

  return (
    <div className="fixed inset-0 z-[110] flex items-center justify-center bg-black/50 backdrop-blur-sm p-4">
      <div className="w-full max-w-3xl rounded-xl bg-white p-8 shadow-2xl animate-in fade-in zoom-in duration-200 overflow-hidden relative">
        
        <div className="flex justify-between items-center mb-8">
          <h2 className="text-2xl font-bold text-gray-900">How HealthRisk Works</h2>
          <button onClick={onClose} className="text-gray-400 hover:text-gray-600 text-2xl">&times;</button>
        </div>

        {/* Steps Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {steps.map((step, idx) => (
            <div key={idx} className="flex gap-4">
              <div className="flex-shrink-0 flex items-center justify-center w-12 h-12 rounded-full bg-teal-100 text-teal-700 font-bold text-lg">
                {step.num}
              </div>
              <div>
                <h3 className="font-bold text-gray-900">{step.title}</h3>
                <p className="text-sm text-gray-600 mt-1 leading-relaxed">{step.desc}</p>
              </div>
            </div>
          ))}
        </div>

        {/* Footer Action */}
        <div className="mt-10 pt-6 border-t flex justify-end gap-3">
          <button 
            onClick={onClose}
            className="px-6 py-2 rounded-md border border-gray-300 text-gray-700 hover:bg-gray-50 font-medium"
          >
            Close
          </button>
          <Link to="/create-profile">
            <Button>Start Your Journey</Button>
          </Link>
        </div>
      </div>
    </div>
  );
};

// --- UPDATED HERO SECTION ---
const Hero = () => {
  const [isLearnMoreOpen, setIsLearnMoreOpen] = useState(false);

  return (
    <section className="bg-gray-50 pt-10 pb-16 lg:pt-20 lg:pb-32">
      <div className="mx-auto grid max-w-screen-2xl grid-cols-1 gap-12 px-6 md:grid-cols-2 md:items-center md:px-10">
        <div className="flex flex-col justify-center">
          <h1 className="text-4xl font-bold tracking-tight text-gray-900 sm:text-5xl 2xl:text-6xl">
            Know Your Health Risk
          </h1>
          <p className="mt-6 max-w-lg text-base text-gray-600 lg:text-lg 2xl:text-xl">
            Personalized predictions & actionable plans tailored just for you.
            Stop guessing and start living healthier today.
          </p>
          <div className="mt-8 flex flex-col gap-4 sm:flex-row sm:items-center">
            <Link to="/create-profile">
              <Button>Get Started</Button>
            </Link>
            {/* Updated Button: Opens the Modal */}
            <Button 
              variant="link" 
              onClick={() => setIsLearnMoreOpen(true)}
            >
              Learn more
            </Button>
          </div>
        </div>
        <div className="flex items-center justify-center">
          <img
            src="https://images.unsplash.com/photo-1512621776951-a57141f2eefd?auto=format&fit=crop&q=80&w=2070"
            alt="Healthy lifestyle"
            className="w-full max-w-xl rounded-xl object-cover shadow-lg lg:max-w-2xl"
          />
        </div>
      </div>

      {/* The Modal lives here inside the section */}
      <LearnMoreModal 
        isOpen={isLearnMoreOpen} 
        onClose={() => setIsLearnMoreOpen(false)} 
      />
    </section>
  );
};

// --- NEW COMPONENT: Feature Modal ---
const FeatureModal = ({ feature, onClose }) => {
  if (!feature) return null;

  return (
    <div className="fixed inset-0 z-[110] flex items-center justify-center bg-black/50 backdrop-blur-sm p-4">
      <div className="w-full max-w-lg rounded-xl bg-white p-8 shadow-2xl animate-in fade-in zoom-in duration-200">
        
        {/* Header with Icon */}
        <div className="text-center">
          <div className="text-6xl mb-4">{feature.icon}</div>
          <h2 className="text-2xl font-bold text-gray-900">{feature.title}</h2>
        </div>

        {/* Detailed Description */}
        <div className="mt-6 text-gray-600 leading-relaxed space-y-4">
          <p>{feature.detail}</p>
          <ul className="list-disc list-inside text-sm bg-gray-50 p-4 rounded-md">
            {feature.bullets.map((item, i) => (
              <li key={i} className="mb-1">{item}</li>
            ))}
          </ul>
        </div>

        {/* Action Buttons */}
        <div className="mt-8 flex flex-col gap-3 sm:flex-row sm:justify-center">
          <Link to="/create-profile" className="w-full sm:w-auto">
            <Button className="w-full">Get Started</Button>
          </Link>
          <button 
            onClick={onClose}
            className="rounded-md border border-gray-300 bg-white px-6 py-3 text-sm font-medium text-gray-700 hover:bg-gray-50 w-full sm:w-auto"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};

// --- UPDATED FEATURES SECTION ---
const Features = () => {
  const [activeFeature, setActiveFeature] = useState(null);

  // Data for the features
  const featuresData = [
    {
      id: 1,
      icon: "❤️",
      title: "Risk Prediction",
      description: "See Obesity & Diabetes risk, not just steps and calories.",
      detail: "Our advanced Machine Learning model analyzes 15+ health markers including BMI, family history, and lifestyle habits to predict your risk of Obesity and Type 2 Diabetes with high accuracy.",
      bullets: ["Instant Analysis", "Color-coded Risk Levels", "Based on Clinical Data"]
    },
    {
      id: 2,
      icon: "🍲",
      title: "Meal Plans",
      description: "AI‑personalized meal plans based on your risk and lifestyle.",
      detail: "Forget generic diets. We generate a 7-day meal plan tailored specifically to lower your identified health risks. Includes recipes, grocery lists, and macro tracking.",
      bullets: ["Risk-Adjusted Calories", "Easy-to-cook Recipes", "One-Click Logging"]
    },
    {
      id: 3,
      icon: "🏋️",
      title: "Exercise Plans",
      description: "Custom routines that match your time and fitness level.",
      detail: "Whether you have 15 minutes or an hour, our algorithm builds a workout schedule that fits your life. We balance cardio, strength, and flexibility to maximize health improvements.",
      bullets: ["Home or Gym Options", "Video Demonstrations", "Progress Tracking"]
    }
  ];

  return (
    <section id="features" className="bg-white py-20 lg:py-32">
      <div className="mx-auto max-w-screen-2xl px-6 md:px-10">
        <h2 className="text-center text-3xl font-bold text-gray-900 mb-16">Why is this app different?</h2>
        
        <div className="grid grid-cols-1 gap-8 sm:grid-cols-2 md:grid-cols-3 lg:gap-12">
          {featuresData.map((feature) => (
            <div 
              key={feature.id}
              onClick={() => setActiveFeature(feature)}
              className="group flex flex-col items-center text-center cursor-pointer rounded-xl p-6 transition-all duration-300 hover:bg-gray-50 hover:shadow-lg hover:-translate-y-1"
            >
              <div className="text-5xl mb-4 transition-transform duration-300 group-hover:scale-110">{feature.icon}</div>
              <h3 className="text-xl font-bold text-gray-900 group-hover:text-teal-600 transition-colors">{feature.title}</h3>
              <p className="mt-2 text-gray-600">{feature.description}</p>
              <span className="mt-4 text-sm font-semibold text-teal-600 opacity-0 transition-opacity group-hover:opacity-100">
                Read More →
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Feature Modal */}
      <FeatureModal 
        feature={activeFeature} 
        onClose={() => setActiveFeature(null)} 
      />
    </section>
  );
};

const About = () => (
  <section id="about" className="bg-gray-50 py-20 lg:py-32">
    <div className="mx-auto max-w-screen-2xl px-6 md:px-10">
      <div className="grid grid-cols-1 gap-12 lg:grid-cols-2 lg:items-center">
        <div>
          <h2 className="text-3xl font-bold text-gray-900 mb-6">Our Mission</h2>
          <p className="text-lg text-gray-600 mb-6 leading-relaxed">
            We believe that understanding your health shouldn't require a medical degree. 
            Our mission is to use advanced Machine Learning to democratize health data, 
            giving you early warnings and clear, actionable steps to prevent chronic diseases.
          </p>
          <p className="text-lg text-gray-600 leading-relaxed">
            Whether you are looking to lose weight, manage glucose levels, or simply live 
            a longer life, our tools adapt to your unique physiology and lifestyle.
          </p>
        </div>
        <div className="relative h-64 lg:h-96 w-full overflow-hidden rounded-xl shadow-lg">
          <img 
            src="https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?auto=format&fit=crop&q=80&w=2070" 
            alt="Medical research team" 
            className="h-full w-full object-cover"
          />
        </div>
      </div>
    </div>
  </section>
);

const Contact = () => (
  <section id="contact" className="bg-white py-20 lg:py-32">
    <div className="mx-auto max-w-screen-md px-6 md:px-10 text-center">
      <h2 className="text-3xl font-bold text-gray-900 mb-4">Get in Touch</h2>
      <p className="text-gray-600 mb-12">Have questions about the methodology or features? Send us a message.</p>
      
      <form className="space-y-6 text-left">
        <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">First Name</label>
            <input type="text" className="w-full rounded-md border border-gray-300 px-4 py-2 focus:border-teal-500 focus:outline-none focus:ring-1 focus:ring-teal-500" placeholder="John" />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Last Name</label>
            <input type="text" className="w-full rounded-md border border-gray-300 px-4 py-2 focus:border-teal-500 focus:outline-none focus:ring-1 focus:ring-teal-500" placeholder="Doe" />
          </div>
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Email</label>
          <input type="email" className="w-full rounded-md border border-gray-300 px-4 py-2 focus:border-teal-500 focus:outline-none focus:ring-1 focus:ring-teal-500" placeholder="john@example.com" />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Message</label>
          <textarea rows="4" className="w-full rounded-md border border-gray-300 px-4 py-2 focus:border-teal-500 focus:outline-none focus:ring-1 focus:ring-teal-500" placeholder="How can we help you?"></textarea>
        </div>
        <div className="text-center">
           <Button className="w-full sm:w-auto px-12">Send Message</Button>
        </div>
      </form>
    </div>
  </section>
);

// --- NEW COMPONENT: Text Modal (For Privacy & Terms) ---
const TextModal = ({ isOpen, onClose, title, content }) => {
  if (!isOpen) return null;
  return (
    <div className="fixed inset-0 z-[110] flex items-center justify-center bg-black/50 backdrop-blur-sm p-4">
      <div className="w-full max-w-2xl max-h-[80vh] flex flex-col rounded-xl bg-white shadow-2xl animate-in fade-in zoom-in duration-200">
        {/* Modal Header */}
        <div className="flex justify-between items-center border-b p-6">
          <h2 className="text-xl font-bold text-gray-900">{title}</h2>
          <button onClick={onClose} className="text-gray-400 hover:text-gray-600 text-2xl">&times;</button>
        </div>
        
        {/* Modal Content (Scrollable) */}
        <div className="overflow-y-auto p-6 text-sm text-gray-600 leading-relaxed space-y-4">
          {content}
        </div>

        {/* Modal Footer */}
        <div className="border-t p-4 text-right bg-gray-50 rounded-b-xl">
          <button 
            onClick={onClose}
            className="rounded-md bg-teal-600 px-6 py-2 text-sm font-medium text-white hover:bg-teal-700"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};

// --- UPDATED FOOTER (With State for Modals) ---
const Footer = () => {
  const [activeModal, setActiveModal] = useState(null); // 'privacy', 'terms', or null

  const privacyContent = (
    <>
      <p><strong>1. Data Collection:</strong> We collect personal health data (weight, height, age) solely for the purpose of generating risk predictions.</p>
      <p><strong>2. Data Security:</strong> Your data is encrypted and stored locally on your device where possible. We do not sell your data to third parties.</p>
      <p><strong>3. Medical Disclaimer:</strong> This application is for informational purposes only and does not constitute medical advice. Always consult a professional.</p>
    </>
  );

  const termsContent = (
    <>
      <p><strong>1. Acceptance:</strong> By using HealthRisk, you agree to these terms.</p>
      <p><strong>2. Usage:</strong> You must be at least 18 years old to use this service.</p>
      <p><strong>3. Liability:</strong> We are not liable for any health decisions made based on the AI predictions provided by this app.</p>
    </>
  );

  return (
    <footer className="border-t bg-gray-50">
      <div className="mx-auto flex max-w-screen-2xl flex-col items-center justify-between gap-6 px-6 py-8 text-sm text-gray-500 md:flex-row md:px-10">
        <span>© 2025 HealthRisk</span>
        <div className="flex gap-6">
          {/* Changed <a> tags to <button> to trigger modals */}
          <button onClick={() => setActiveModal('privacy')} className="hover:text-teal-700 hover:underline">
            Privacy Policy
          </button>
          <button onClick={() => setActiveModal('terms')} className="hover:text-teal-700 hover:underline">
            Terms
          </button>
          <a href="#contact" className="hover:text-teal-700 hover:underline">
            Contact
          </a>
        </div>
      </div>

      {/* The Modals live here */}
      <TextModal 
        isOpen={activeModal === 'privacy'} 
        onClose={() => setActiveModal(null)} 
        title="Privacy Policy"
        content={privacyContent}
      />
      <TextModal 
        isOpen={activeModal === 'terms'} 
        onClose={() => setActiveModal(null)} 
        title="Terms of Service"
        content={termsContent}
      />
    </footer>
  );
};

// --- Updated HomePage Wrapper (Handles Modal State) ---
const HomePage = () => {
  const [isLoginOpen, setIsLoginOpen] = useState(false);

  return (
    <div className="flex min-h-screen flex-col font-sans text-gray-900 scroll-smooth">
      <Header onLoginClick={() => setIsLoginOpen(true)} />
      <main className="flex-grow flex flex-col">
        <Hero />
        <Features />
        <About />
        <Contact />
      </main>
      <Footer />
      
      {/* The Login Popup */}
      <LoginModal isOpen={isLoginOpen} onClose={() => setIsLoginOpen(false)} />
    </div>
  );
};

// --- Main App ---
export default function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/create-profile" element={<CreateProfile />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/meal-plan" element={<MealPlan />} />
        <Route path="/exercise-plan" element={<ExercisePlan />} />
        <Route path="/progress" element={<Progress />} />
      </Routes>
    </Router>
  );
}