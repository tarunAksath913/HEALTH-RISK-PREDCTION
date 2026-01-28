import React, { useState, useEffect, useRef } from "react";
import { Link, useLocation } from "react-router-dom";

// --- Icons ---
const Icons = {
  Bell: () => <svg className="w-5 h-5 text-gray-500 group-hover:text-gray-700" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9" /></svg>,
  BellOutline: () => <svg className="w-6 h-6 text-gray-500 hover:text-gray-700 transition" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9" /></svg>,
  Settings: () => <svg className="w-6 h-6 text-gray-500 hover:text-gray-700 transition cursor-pointer" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" /></svg>,
  Meal: () => <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M3 3h2l.4 2M7 13h10l4-8H5.4M7 13L5.4 5M7 13l-2.293 2.293c-.63.63-.184 1.707.707 1.707H17m0 0a2 2 0 100 4 2 2 0 000-4zm-8 2a2 2 0 11-4 0 2 2 0 014 0z" /></svg>,
  Run: () => <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>,
  Chart: () => <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" /></svg>,
  Fire: () => <svg className="w-5 h-5 text-orange-500" fill="currentColor" viewBox="0 0 20 20"><path fillRule="evenodd" d="M12.395 2.553a1 1 0 00-1.45-.385c-.345.23-.614.558-.822.88-.214.33-.403.713-.57 1.116-.334.804-.614 1.768-.84 2.734a31.365 31.365 0 00-.613 3.58 2.64 2.64 0 01-.945-1.067c-.328-.68-.398-1.534-.398-2.654A1 1 0 005.05 6.05 6.981 6.981 0 003 11a7 7 0 1011.95-4.95c-.592-.591-.98-.985-1.348-1.467-.363-.476-.724-1.063-1.207-2.03zM12.12 15.12A3 3 0 017 13s.879.5 2.5.5c0-1 .5-4 1.25-4.5.5 1 .786 1.293 1.371 1.879A2.99 2.99 0 0113 13a2.99 2.99 0 01-.879 2.121z" clipRule="evenodd" /></svg>,
  User: () => <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" /></svg>,
  Moon: () => <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" /></svg>,
  Help: () => <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>,
  Logout: () => <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" /></svg>,
  Camera: () => <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" /></svg>,
};

// --- Helper: Convert Obesity String to Dashboard Visuals ---
const getObesityData = (levelString) => {
  // Default fallback
  if (!levelString) return { score: 50, category: "Unknown", colorParams: { bg: "bg-gray-100", badge: "bg-gray-600", progress: "bg-gray-500" } };
  
  // Mapping AI output to UI colors and score
  const map = {
    'Insufficient_Weight': { score: 20, category: "Underweight", colorParams: { bg: "bg-blue-100", badge: "bg-blue-600", progress: "bg-blue-500" } },
    'Normal_Weight':       { score: 5,  category: "Normal",      colorParams: { bg: "bg-green-100", badge: "bg-green-600", progress: "bg-green-500" } },
    'Overweight_Level_I':  { score: 45, category: "Overweight I",colorParams: { bg: "bg-yellow-100", badge: "bg-yellow-600", progress: "bg-yellow-500" } },
    'Overweight_Level_II': { score: 60, category: "Overweight II",colorParams: { bg: "bg-orange-100", badge: "bg-orange-600", progress: "bg-orange-500" } },
    'Obesity_Type_I':      { score: 75, category: "Obesity I",   colorParams: { bg: "bg-red-100", badge: "bg-red-600", progress: "bg-red-500" } },
    'Obesity_Type_II':     { score: 90, category: "Obesity II",  colorParams: { bg: "bg-rose-100", badge: "bg-rose-600", progress: "bg-rose-500" } },
    'Obesity_Type_III':    { score: 100,category: "Severe",      colorParams: { bg: "bg-purple-100", badge: "bg-purple-600", progress: "bg-purple-500" } },
  };

  return map[levelString] || map['Normal_Weight'];
};

const getDiabetesData = (riskScore) => {
  const score = parseFloat(riskScore) || 0;
  if (score < 30) return { category: "Low Risk", colorParams: { bg: "bg-emerald-100", badge: "bg-emerald-600", progress: "bg-emerald-500" }, recs: ["Maintain healthy diet", "Yearly check-up"] };
  if (score < 60) return { category: "Moderate", colorParams: { bg: "bg-yellow-100", badge: "bg-yellow-600", progress: "bg-yellow-500" }, recs: ["Reduce sugar intake", "Increase cardio", "Monitor glucose"] };
  return { category: "High Risk", colorParams: { bg: "bg-red-100", badge: "bg-red-600", progress: "bg-red-500" }, recs: ["Consult a doctor", "Strict diet plan", "Daily exercise"] };
};

// --- Mock Notifications ---
const MOCK_NOTIFICATIONS = [
  { id: 1, text: "🎉 Congrats! You reached a 5-day streak.", time: "2 min ago", unread: true },
  { id: 2, text: "🍽️ Don't forget to log your lunch.", time: "1 hour ago", unread: true },
  { id: 3, text: "💪 New Strength Workout added.", time: "Yesterday", unread: false },
];

// --- Components ---

const ProfileEditModal = ({ isOpen, onClose }) => {
  if (!isOpen) return null;
  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/50 backdrop-blur-sm p-4">
      <div className="w-full max-w-md rounded-xl bg-white p-6 shadow-2xl animate-in fade-in zoom-in duration-200">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-xl font-bold text-gray-900">Edit Profile</h2>
          <button onClick={onClose} className="text-gray-400 hover:text-gray-600 text-2xl">&times;</button>
        </div>
        <form onSubmit={(e) => { e.preventDefault(); onClose(); }} className="space-y-4">
          <div className="flex flex-col items-center gap-2 mb-4">
            <div className="relative group cursor-pointer">
              <div className="h-20 w-20 rounded-full bg-teal-100 border-2 border-teal-200 flex items-center justify-center text-teal-700 font-bold text-2xl">JD</div>
              <div className="absolute inset-0 rounded-full bg-black/40 flex items-center justify-center opacity-0 group-hover:opacity-100 transition text-white"><Icons.Camera /></div>
            </div>
            <span className="text-xs text-teal-600 font-medium cursor-pointer">Change Photo</span>
          </div>
          <div><label className="block text-sm font-medium text-gray-700 mb-1">Full Name</label><input type="text" defaultValue="John Doe" className="w-full rounded-md border border-gray-300 px-3 py-2 focus:outline-none focus:ring-1 focus:ring-teal-500" /></div>
          <div><label className="block text-sm font-medium text-gray-700 mb-1">Email</label><input type="email" defaultValue="john.doe@example.com" className="w-full rounded-md border border-gray-300 px-3 py-2 focus:outline-none focus:ring-1 focus:ring-teal-500" /></div>
          <div><label className="block text-sm font-medium text-gray-700 mb-1">New Password</label><input type="password" placeholder="Leave blank to keep current" className="w-full rounded-md border border-gray-300 px-3 py-2 focus:outline-none focus:ring-1 focus:ring-teal-500" /></div>
          <div className="pt-4 flex gap-3">
             <button type="button" onClick={onClose} className="flex-1 rounded-md border border-gray-300 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50">Cancel</button>
             <button type="submit" className="flex-1 rounded-md bg-teal-600 py-2 text-sm font-medium text-white hover:bg-teal-700">Save Changes</button>
          </div>
        </form>
      </div>
    </div>
  );
};

const DashboardHeader = () => {
  const [showNotifications, setShowNotifications] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [showProfileEdit, setShowProfileEdit] = useState(false);
  const [hasUnread, setHasUnread] = useState(true);
  const [darkMode, setDarkMode] = useState(false);
  const [notificationsEnabled, setNotificationsEnabled] = useState(true);
  const notifRef = useRef(null);
  const settingsRef = useRef(null);

  const handleBellClick = () => { setShowNotifications(!showNotifications); setShowSettings(false); if (hasUnread) setHasUnread(false); };
  const handleSettingsClick = () => { setShowSettings(!showSettings); setShowNotifications(false); };
  const handleEditProfileClick = () => { setShowProfileEdit(true); setShowSettings(false); };

  useEffect(() => {
    function handleClickOutside(event) {
      if (notifRef.current && !notifRef.current.contains(event.target)) setShowNotifications(false);
      if (settingsRef.current && !settingsRef.current.contains(event.target)) setShowSettings(false);
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  return (
    <>
      <header className="sticky top-0 z-30 bg-white border-b px-6 py-4">
        <div className="mx-auto flex max-w-6xl items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="h-8 w-8 bg-teal-600 rounded-lg flex items-center justify-center text-white font-bold">H</div>
            <span className="text-lg font-bold text-gray-900 hidden sm:block">HealthRisk</span>
          </div>
          <div className="flex items-center gap-6">
            <div className="relative" ref={notifRef}>
              <button onClick={handleBellClick} className="relative p-1 rounded-full hover:bg-gray-100 transition focus:outline-none">
                <Icons.BellOutline />
                {hasUnread && notificationsEnabled && <span className="absolute top-1 right-1 h-2.5 w-2.5 bg-red-500 rounded-full border border-white"></span>}
              </button>
              {showNotifications && (
                <div className="absolute right-0 mt-3 w-80 bg-white rounded-xl shadow-xl border border-gray-100 overflow-hidden animate-in fade-in slide-in-from-top-2 z-50">
                  <div className="px-4 py-3 border-b bg-gray-50 flex justify-between items-center"><span className="text-sm font-bold text-gray-800">Notifications</span><span className="text-xs text-teal-600 cursor-pointer hover:underline">Mark all read</span></div>
                  {notificationsEnabled ? (
                    <ul className="max-h-64 overflow-y-auto">{MOCK_NOTIFICATIONS.map((notif) => (<li key={notif.id} className={`px-4 py-3 border-b hover:bg-gray-50 transition cursor-pointer ${notif.unread ? "bg-blue-50/50" : ""}`}><p className="text-sm text-gray-800">{notif.text}</p><p className="text-xs text-gray-400 mt-1">{notif.time}</p></li>))}</ul>
                  ) : (<div className="p-6 text-center text-sm text-gray-500">Notifications paused.</div>)}
                </div>
              )}
            </div>
            <div className="relative" ref={settingsRef}>
              <button onClick={handleSettingsClick} className="p-1 rounded-full hover:bg-gray-100 transition focus:outline-none"><Icons.Settings /></button>
              {showSettings && (
                <div className="absolute right-0 mt-3 w-64 bg-white rounded-xl shadow-xl border border-gray-100 overflow-hidden animate-in fade-in slide-in-from-top-2 z-50">
                  <div className="px-4 py-3 border-b bg-gray-50"><p className="text-sm font-bold text-gray-900">Settings</p></div>
                  <div className="p-2 space-y-1">
                    <button onClick={handleEditProfileClick} className="w-full flex items-center gap-3 px-3 py-2 text-sm text-gray-700 hover:bg-gray-50 rounded-md transition"><Icons.User /><span>Edit Profile</span></button>
                    <div onClick={() => setDarkMode(!darkMode)} className="flex items-center justify-between px-3 py-2 text-sm text-gray-700 hover:bg-gray-50 rounded-md transition cursor-pointer"><div className="flex items-center gap-3"><Icons.Moon /><span>Dark Theme</span></div><div className={`w-9 h-5 rounded-full relative transition-colors ${darkMode ? 'bg-teal-600' : 'bg-gray-300'}`}><div className={`absolute top-0.5 w-4 h-4 bg-white rounded-full transition-transform ${darkMode ? 'left-4.5 translate-x-0.5' : 'left-0.5'}`}></div></div></div>
                    <div onClick={() => setNotificationsEnabled(!notificationsEnabled)} className="flex items-center justify-between px-3 py-2 text-sm text-gray-700 hover:bg-gray-50 rounded-md transition cursor-pointer"><div className="flex items-center gap-3"><Icons.Bell /><span>Notifications</span></div><div className={`w-9 h-5 rounded-full relative transition-colors ${notificationsEnabled ? 'bg-teal-600' : 'bg-gray-300'}`}><div className={`absolute top-0.5 w-4 h-4 bg-white rounded-full transition-transform ${notificationsEnabled ? 'left-4.5 translate-x-0.5' : 'left-0.5'}`}></div></div></div>
                    <a href="#" className="flex items-center gap-3 px-3 py-2 text-sm text-gray-700 hover:bg-gray-50 rounded-md transition"><Icons.Help /><span>Help & Support</span></a>
                  </div>
                  <div className="border-t p-2"><Link to="/" className="flex items-center gap-3 px-3 py-2 text-sm text-red-600 hover:bg-red-50 rounded-md transition font-medium"><Icons.Logout /><span>Log Out</span></Link></div>
                </div>
              )}
            </div>
            <div onClick={handleSettingsClick} className="flex items-center gap-3 pl-4 border-l cursor-pointer hover:opacity-80 transition">
              <div className="text-right hidden md:block"><p className="text-sm font-semibold text-gray-900">John Doe</p><p className="text-xs text-gray-500">Free Plan</p></div>
              <div className="h-10 w-10 rounded-full bg-teal-100 border border-teal-200 flex items-center justify-center text-teal-700 font-bold">JD</div>
            </div>
          </div>
        </div>
      </header>
      <ProfileEditModal isOpen={showProfileEdit} onClose={() => setShowProfileEdit(false)} />
    </>
  );
};

const StatCard = ({ label, value, subtext, icon }) => (
  <div className="bg-white p-5 rounded-xl shadow-sm border border-gray-100 flex items-center justify-between">
    <div><p className="text-xs font-semibold text-gray-400 uppercase tracking-wider">{label}</p><p className="text-2xl font-bold text-gray-900 mt-1">{value}</p><p className="text-xs text-teal-600 font-medium mt-1">{subtext}</p></div>
    <div className="h-10 w-10 bg-gray-50 rounded-full flex items-center justify-center text-gray-400">{icon}</div>
  </div>
);

const RiskCard = ({ title, score, category, colorParams, recommendation }) => {
  const [animatedScore, setAnimatedScore] = useState(0);
  useEffect(() => {
    let start = 0;
    const end = score;
    const duration = 2000;
    const incrementTime = 20;
    const totalSteps = duration / incrementTime;
    const incrementValue = end / totalSteps;
    const timer = setInterval(() => { start += incrementValue; if (start >= end) { setAnimatedScore(end); clearInterval(timer); } else { setAnimatedScore(Math.floor(start)); } }, incrementTime);
    return () => clearInterval(timer);
  }, [score]);

  return (
    <div className="overflow-hidden rounded-xl bg-white shadow-sm border border-gray-100 transition hover:shadow-md">
      <div className={`${colorParams.bg} p-6 text-gray-900 relative`}>
        <div className="flex items-start justify-between relative z-10"><h3 className="text-sm font-bold uppercase tracking-wide opacity-90">{title}</h3><span className={`rounded-full px-3 py-1 text-xs font-bold uppercase text-white ${colorParams.badge} shadow-sm`}>{category}</span></div>
        <div className="mt-4 flex items-baseline gap-1 relative z-10"><span className="text-5xl font-extrabold text-gray-900">{animatedScore}%</span></div>
        <div className="relative mt-4 h-2 w-full rounded-full bg-white/40 z-10 overflow-hidden"><div className={`absolute left-0 top-0 h-full rounded-full ${colorParams.progress} transition-all ease-out duration-[2000ms]`} style={{ width: `${animatedScore}%` }} /></div>
      </div>
      <div className="bg-white p-6"><h4 className="font-semibold text-gray-900 text-sm">Action Plan</h4><ul className="mt-3 space-y-2">{recommendation.map((rec, index) => (<li key={index} className="flex items-center gap-2 text-xs text-gray-600"><div className="w-1.5 h-1.5 rounded-full bg-teal-500"></div>{rec}</li>))}</ul></div>
    </div>
  );
};

const ActionCard = ({ title, desc, icon, to, colorClass }) => (
  <Link to={to} className="group block h-full">
    <div className="h-full bg-white p-6 rounded-xl shadow-sm border border-gray-100 transition duration-300 hover:border-teal-500 hover:shadow-md hover:-translate-y-1">
      <div className={`w-12 h-12 rounded-lg ${colorClass} flex items-center justify-center mb-4 transition-transform group-hover:scale-110`}>{icon}</div>
      <h3 className="text-lg font-bold text-gray-900 group-hover:text-teal-600 transition-colors">{title}</h3><p className="mt-2 text-sm text-gray-500 leading-relaxed">{desc}</p>
      <div className="mt-4 flex items-center text-sm font-medium text-teal-600 opacity-0 transform translate-y-2 transition-all group-hover:opacity-100 group-hover:translate-y-0">View Details →</div>
    </div>
  </Link>
);

// --- Main Page ---
export default function Dashboard() {
  const location = useLocation();
  const [aiData, setAiData] = useState(null);

  useEffect(() => {
    // 1. Get prediction from Router State
    if (location.state && location.state.prediction) {
      const pred = location.state.prediction;
      
      // 2. Process Data for UI
      const obesityMeta = getObesityData(pred.obesity?.obesity_level);
      const diabetesMeta = getDiabetesData(pred.diabetes?.risk_score);

      setAiData({
        diabetes: {
          score: pred.diabetes?.risk_score || 0,
          ...diabetesMeta
        },
        obesity: {
          rawLevel: pred.obesity?.obesity_level,
          ...obesityMeta
        }
      });
    }
  }, [location]);

  // Fallback if no data (e.g., user went straight to /dashboard)
  const displayData = aiData || {
    diabetes: { score: 65, category: "Moderate (Demo)", colorParams: { bg: "bg-orange-100", badge: "bg-orange-600", progress: "bg-orange-500" }, recs: ["Demo Mode: Calculate Profile first"] },
    obesity: { score: 40, category: "Normal (Demo)", colorParams: { bg: "bg-blue-100", badge: "bg-blue-600", progress: "bg-blue-500" }, recs: ["Demo Mode: Calculate Profile first"] }
  };

  return (
    <div className="min-h-screen bg-gray-50 font-sans">
      <DashboardHeader />
      <main className="mx-auto max-w-6xl px-6 py-8">
        <div className="mb-8 animate-in fade-in slide-in-from-bottom-4 duration-700">
          <h1 className="text-2xl font-bold text-gray-900">Good Morning, John! 👋</h1>
          <p className="text-sm text-gray-500">{aiData ? "Here is your personalized AI analysis." : "Please calculate your risk profile to see real data."}</p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-6 mb-8 animate-in fade-in slide-in-from-bottom-6 duration-1000">
          <StatCard label="Current Weight" value="75 kg" subtext="-2kg this month" icon="⚖️" />
          <StatCard label="Daily Calories" value="1,240" subtext="of 2,000 goal" icon={<Icons.Fire />} />
          <StatCard label="Login Streak" value="5 Days" subtext="Keep it up!" icon="🏆" />
        </div>

        <div className="mb-10 animate-in fade-in slide-in-from-bottom-8 duration-1000 delay-100">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-bold text-gray-900">Risk Analysis</h2>
            {!aiData && <Link to="/create-profile" className="text-sm text-teal-600 font-medium hover:underline">Calculate Now</Link>}
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <RiskCard 
              title="Obesity Level"
              score={displayData.obesity.score}
              category={displayData.obesity.category}
              colorParams={displayData.obesity.colorParams}
              recommendation={["Maintain balanced diet", "Regular cardio (150m/week)", "Monitor daily caloric intake"]}
            />
            <RiskCard 
              title="Diabetes Risk"
              score={displayData.diabetes.score}
              category={displayData.diabetes.category}
              colorParams={displayData.diabetes.colorParams}
              recommendation={displayData.diabetes.recs}
            />
          </div>
        </div>

        <div className="animate-in fade-in slide-in-from-bottom-10 duration-1000 delay-200">
          <h2 className="text-lg font-bold text-gray-900 mb-4">Your Tools</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <ActionCard to="/meal-plan" title="Meal Plan" desc="View your daily personalized recipes and log your intake." icon={<Icons.Meal />} colorClass="bg-orange-100 text-orange-600" />
            <ActionCard to="/exercise-plan" title="Exercise Plan" desc="Track your daily workouts and watch technique videos." icon={<Icons.Run />} colorClass="bg-blue-100 text-blue-600" />
            <ActionCard to="/progress" title="View Progress" desc="Analyze your health trends and weight loss journey." icon={<Icons.Chart />} colorClass="bg-purple-100 text-purple-600" />
          </div>
        </div>
      </main>
    </div>
  );
}