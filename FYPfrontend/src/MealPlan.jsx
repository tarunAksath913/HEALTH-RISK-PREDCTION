import React, { useState, useEffect, useRef } from "react";
import { Link } from "react-router-dom";

// --- Icons ---
const Icons = {
  Bell: () => <svg className="w-6 h-6 text-gray-500 hover:text-gray-700 transition" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9" /></svg>,
  Settings: () => <svg className="w-6 h-6 text-gray-500 hover:text-gray-700 transition cursor-pointer" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" /></svg>,
  Moon: () => <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" /></svg>,
  Logout: () => <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" /></svg>,
  ArrowLeft: () => <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 19l-7-7m0 0l7-7m-7 7h18" /></svg>,
  Clock: () => <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>,
  Info: () => <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>,
  Fire: () => <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M17.657 18.657A8 8 0 016.343 7.343S7 9 9 10c0-2 .5-5 2.986-7C14 5 16.09 5.777 17.656 7.343A7.975 7.975 0 0120 13a7.975 7.975 0 01-2.343 5.657z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9.879 16.121A3 3 0 1012.015 11L11 14H9c0 .768.293 1.536.879 2.121z" /></svg>,
  Printer: () => <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M17 17h2a2 2 0 002-2v-4a2 2 0 00-2-2H5a2 2 0 00-2 2v4a2 2 0 002 2h2m2 4h6a2 2 0 002-2v-4a2 2 0 00-2-2H9a2 2 0 00-2 2v4a2 2 0 002 2zm8-12V5a2 2 0 00-2-2H9a2 2 0 00-2 2v4h10z" /></svg>,
  Refresh: () => <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" /></svg>,
  Edit: () => <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" /></svg>,
  Sparkles: () => <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z" /></svg>,
  Calendar: () => <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" /></svg>,
  ViewList: () => <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 6h16M4 12h16M4 18h16" /></svg>
};

// --- Mock Data ---
const DAILY_MEALS = [
  {
    id: 1,
    type: "Breakfast",
    time: "8:00 am",
    title: "Oatmeal with Berries",
    calories: "320 cal",
    protein: "8g Protein",
    img: "🥣",
    ingredients: ["1/2 cup rolled oats", "1.5g cinnamon", "1 tbsp chia seeds", "1 cup almond milk", "1/2 cup blueberries"],
    steps: ["Pour oats and almond milk into a bowl.", "Microwave for 2 minutes or cook on stovetop.", "Stir in cinnamon and chia seeds.", "Top with fresh blueberries and serve warm."]
  },
  {
    id: 2,
    type: "Lunch",
    time: "12:30 pm",
    title: "Grilled Chicken Salad",
    calories: "450 cal",
    protein: "40g Protein",
    img: "🥗",
    ingredients: ["150g Chicken breast", "2 cups mixed greens", "1 tbsp olive oil", "1/2 cup cherry tomatoes", "1/4 cucumber, sliced"],
    steps: ["Season chicken breast with salt, pepper, and herbs.", "Grill chicken until cooked through (approx 6 mins per side).", "Chop vegetables and place in a bowl.", "Slice chicken and place on top.", "Drizzle with olive oil."]
  },
  {
    id: 3,
    type: "Dinner",
    time: "7:00 pm",
    title: "Baked Salmon & Veggies",
    calories: "500 cal",
    protein: "35g Protein",
    img: "🐟",
    ingredients: ["1 Salmon fillet (150g)", "1 cup broccoli florets", "1/2 cup quinoa (cooked)", "1 slice lemon", "1 tsp garlic powder"],
    steps: ["Preheat oven to 180°C (350°F).", "Place salmon and broccoli on a baking sheet.", "Season with garlic powder, salt, and lemon juice.", "Bake for 15-20 minutes.", "Serve with a side of cooked quinoa."]
  }
];

const WEEKLY_PLAN = [
    { day: "Monday", b: "Oatmeal with Berries", l: "Grilled Chicken Salad", d: "Baked Salmon & Veggies" },
    { day: "Tuesday", b: "Greek Yogurt Parfait", l: "Turkey Wrap", d: "Stir-fry Tofu & Veggies" },
    { day: "Wednesday", b: "Avocado Toast", l: "Quinoa Salad", d: "Lean Beef Steak" },
    { day: "Thursday", b: "Smoothie Bowl", l: "Chicken Soup", d: "Whole Wheat Pasta" },
    { day: "Friday", b: "Scrambled Eggs", l: "Tuna Salad", d: "Roasted Chicken" },
    { day: "Saturday", b: "Pancakes (Oat flour)", l: "Burrito Bowl", d: "Fish Tacos" },
    { day: "Sunday", b: "Veggie Omelet", l: "Lentil Soup", d: "Turkey Meatballs" }
];

// --- Sub-Components ---
const Toast = ({ show }) => (
  <div 
    className={`fixed top-24 right-6 z-[120] flex items-center gap-3 rounded-lg bg-emerald-600 px-4 py-3 text-white shadow-xl transition-all duration-300 ${
      show ? "translate-x-0 opacity-100" : "translate-x-10 opacity-0 pointer-events-none"
    }`}
  >
    <div className="text-xl">✅</div>
    <div>
      <p className="font-bold text-sm">Success</p>
      <p className="text-xs opacity-90">Meal logged to your daily total!</p>
    </div>
  </div>
);

const RecipeModal = ({ recipe, onClose }) => {
  if (!recipe) return null;

  return (
    <div className="fixed inset-0 z-[110] flex items-center justify-center bg-black/60 backdrop-blur-sm p-4 animate-in fade-in duration-200">
      <div className="w-full max-w-lg rounded-xl bg-white shadow-2xl overflow-hidden animate-in zoom-in-95 duration-200">
        <div className="bg-teal-600 px-6 py-4 flex justify-between items-center text-white">
            <h2 className="text-xl font-bold">{recipe.title}</h2>
            <button onClick={onClose} className="text-teal-100 hover:text-white text-2xl font-bold">&times;</button>
        </div>
        <div className="p-6 max-h-[80vh] overflow-y-auto">
            <div className="flex items-center justify-between mb-6">
                <span className="inline-block rounded-full bg-teal-50 px-3 py-1 text-xs font-bold text-teal-700 uppercase tracking-wide border border-teal-100">
                    {recipe.type}
                </span>
                <div className="flex gap-4 text-sm text-gray-500 font-medium">
                    <span className="flex items-center gap-1 text-orange-600"><Icons.Fire /> {recipe.calories}</span>
                    <span className="flex items-center gap-1 text-blue-600">💪 {recipe.protein}</span>
                </div>
            </div>
            <div className="mb-6">
                <h3 className="font-bold text-gray-900 text-sm uppercase tracking-wide border-b pb-2 mb-3">Ingredients</h3>
                <ul className="space-y-2">
                    {recipe.ingredients.map((ing, i) => (
                        <li key={i} className="flex items-center gap-2 text-sm text-gray-700">
                            <div className="w-1.5 h-1.5 rounded-full bg-teal-500"></div>
                            {ing}
                        </li>
                    ))}
                </ul>
            </div>
            <div className="mb-6">
                <h3 className="font-bold text-gray-900 text-sm uppercase tracking-wide border-b pb-2 mb-3">Directions</h3>
                <ol className="space-y-3">
                    {recipe.steps.map((step, i) => (
                        <li key={i} className="flex gap-3 text-sm text-gray-700">
                            <span className="flex-shrink-0 w-5 h-5 rounded-full bg-gray-100 text-gray-500 text-xs font-bold flex items-center justify-center">{i + 1}</span>
                            {step}
                        </li>
                    ))}
                </ol>
            </div>
        </div>
        <div className="bg-gray-50 px-6 py-4 border-t text-right">
            <button onClick={onClose} className="rounded-md border border-gray-300 bg-white px-6 py-2 text-sm font-medium text-gray-700 hover:bg-gray-100 transition shadow-sm">
                Close
            </button>
        </div>
      </div>
    </div>
  );
};

// --- Shared Header ---
const SharedHeader = () => {
  const [showNotifications, setShowNotifications] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [darkMode, setDarkMode] = useState(false);
  
  const notifRef = useRef(null);
  const settingsRef = useRef(null);

  useEffect(() => {
    function handleClickOutside(event) {
      if (notifRef.current && !notifRef.current.contains(event.target)) setShowNotifications(false);
      if (settingsRef.current && !settingsRef.current.contains(event.target)) setShowSettings(false);
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  return (
    <header className="sticky top-0 z-30 bg-white border-b px-6 py-4 print:hidden">
      <div className="mx-auto flex max-w-6xl items-center justify-between">
        <Link to="/dashboard" className="flex items-center gap-2 group">
          <div className="h-8 w-8 bg-teal-600 rounded-lg flex items-center justify-center text-white font-bold group-hover:bg-teal-700 transition">H</div>
          <span className="text-lg font-bold text-gray-900 hidden sm:block">HealthRisk</span>
        </Link>
        <div className="flex items-center gap-6">
          <div className="relative" ref={notifRef}>
            <button onClick={() => setShowNotifications(!showNotifications)} className="relative p-1 rounded-full hover:bg-gray-100 transition"><Icons.Bell /></button>
            {showNotifications && (
              <div className="absolute right-0 mt-3 w-80 bg-white rounded-xl shadow-xl border border-gray-100 overflow-hidden animate-in fade-in slide-in-from-top-2 z-50">
                 <div className="px-4 py-3 border-b bg-gray-50 font-bold text-sm">Notifications</div>
                 <ul className="max-h-64 overflow-y-auto">
                    <li className="px-4 py-3 border-b text-sm text-gray-600 hover:bg-gray-50">Meal Plan Updated</li>
                 </ul>
              </div>
            )}
          </div>
          <div className="relative" ref={settingsRef}>
            <button onClick={() => setShowSettings(!showSettings)} className="p-1 rounded-full hover:bg-gray-100 transition"><Icons.Settings /></button>
            {showSettings && (
              <div className="absolute right-0 mt-3 w-64 bg-white rounded-xl shadow-xl border border-gray-100 overflow-hidden animate-in fade-in slide-in-from-top-2 z-50">
                 <div className="px-4 py-3 border-b bg-gray-50">
                   <p className="text-sm font-bold text-gray-900">Settings</p>
                 </div>
                 <div className="p-2 space-y-1">
                   {/* Dark Mode Toggle */}
                   <div 
                      onClick={() => setDarkMode(!darkMode)} 
                      className="flex items-center justify-between px-3 py-2 text-sm text-gray-700 hover:bg-gray-50 rounded-md transition cursor-pointer"
                    >
                      <div className="flex items-center gap-3">
                        <Icons.Moon />
                        <span>Dark Theme</span>
                      </div>
                      <div className={`w-9 h-5 rounded-full relative transition-colors duration-300 ${darkMode ? 'bg-teal-600' : 'bg-gray-300'}`}>
                        <div className={`absolute top-0.5 w-4 h-4 bg-white rounded-full transition-transform duration-300 shadow-sm ${darkMode ? 'left-4.5 translate-x-0.5' : 'left-0.5'}`}></div>
                      </div>
                    </div>
                 </div>
                 <div className="border-t p-2">
                   <Link to="/" className="flex items-center gap-3 px-3 py-2 text-sm text-red-600 hover:bg-red-50 rounded-md transition font-medium">
                     <Icons.Logout />
                     <span>Log Out</span>
                   </Link>
                 </div>
              </div>
            )}
          </div>
          <div className="flex items-center gap-3 pl-4 border-l">
            <div className="h-10 w-10 rounded-full bg-teal-100 border border-teal-200 flex items-center justify-center text-teal-700 font-bold">JD</div>
          </div>
        </div>
      </div>
    </header>
  );
};

// --- Main Page Component ---
export default function MealPlan() {
  const [selectedRecipe, setSelectedRecipe] = useState(null);
  const [showToast, setShowToast] = useState(false);
  const [viewMode, setViewMode] = useState("daily"); // 'daily' or 'weekly'

  const handleLogItem = () => {
    setShowToast(true);
    setTimeout(() => setShowToast(false), 3000);
  };

  const handlePrint = () => {
    window.print();
  };

  return (
    <div className="min-h-screen bg-gray-50 font-sans">
      <SharedHeader />

      <main className="mx-auto max-w-6xl px-6 py-8">
        <Toast show={showToast} />
        <RecipeModal recipe={selectedRecipe} onClose={() => setSelectedRecipe(null)} />

        {/* Page Header with View Toggle */}
        <div className="mb-8 flex flex-col md:flex-row md:items-end md:justify-between gap-4">
          <div>
            <Link to="/dashboard" className="flex items-center gap-2 text-sm text-gray-500 hover:text-teal-600 mb-2 transition print:hidden">
              <Icons.ArrowLeft /> Back to Dashboard
            </Link>
            <h1 className="text-2xl font-bold text-gray-900">
               {viewMode === "daily" ? "Today's Meal Plan" : "Weekly Overview"}
            </h1>
            <div className="flex items-center gap-1 text-xs font-medium text-purple-600 mt-1">
              <Icons.Sparkles /> AI Based Generation
            </div>
          </div>
          
          {/* Action Buttons & Toggle */}
          <div className="flex flex-col md:flex-row md:items-center gap-4 print:hidden">
             
             {/* View Toggle */}
             <div className="flex rounded-lg bg-white p-1 shadow-sm border border-gray-200">
                <button 
                  onClick={() => setViewMode("daily")}
                  className={`flex items-center gap-2 px-4 py-1.5 rounded-md text-sm font-medium transition ${viewMode === "daily" ? "bg-teal-100 text-teal-700" : "text-gray-500 hover:text-gray-700"}`}
                >
                  <Icons.ViewList /> Day
                </button>
                <button 
                  onClick={() => setViewMode("weekly")}
                  className={`flex items-center gap-2 px-4 py-1.5 rounded-md text-sm font-medium transition ${viewMode === "weekly" ? "bg-teal-100 text-teal-700" : "text-gray-500 hover:text-gray-700"}`}
                >
                  <Icons.Calendar /> Week
                </button>
             </div>

             <div className="w-px h-8 bg-gray-300 hidden md:block"></div>

             <div className="flex gap-2">
                <button className="flex items-center gap-2 px-4 py-2 bg-white border border-gray-200 rounded-lg text-sm font-medium text-gray-700 hover:bg-gray-50 shadow-sm transition">
                  <Icons.Edit /> Edit
                </button>
                <button className="flex items-center gap-2 px-4 py-2 bg-white border border-gray-200 rounded-lg text-sm font-medium text-gray-700 hover:bg-gray-50 shadow-sm transition">
                  <Icons.Refresh /> Regenerate
                </button>
                <button onClick={handlePrint} className="flex items-center gap-2 px-4 py-2 bg-teal-600 border border-transparent rounded-lg text-sm font-medium text-white hover:bg-teal-700 shadow-sm transition">
                  <Icons.Printer /> Print
                </button>
             </div>
          </div>
        </div>

        {/* DAILY VIEW CONTENT */}
        {viewMode === "daily" && (
          <>
            {/* Daily Target Banner */}
            <div className="mb-8 rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 px-6 py-6 text-white shadow-md print:shadow-none print:border print:border-gray-200 print:text-black print:bg-none">
              <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
                <div>
                  <h3 className="font-bold text-lg flex items-center gap-2"><Icons.Info /> Daily Targets</h3>
                  <p className="text-blue-100 text-sm mt-1 print:text-gray-600">Goal: 1,800 calories • Macros: 50% Carbs, 30% Protein, 20% Fat</p>
                </div>
                <div className="flex gap-4 text-center">
                  <div>
                    <span className="block text-2xl font-bold">1,240</span>
                    <span className="text-xs text-blue-200 uppercase tracking-wide print:text-gray-500">Eaten</span>
                  </div>
                  <div className="w-px bg-blue-400/30 print:bg-gray-300"></div>
                  <div>
                    <span className="block text-2xl font-bold">560</span>
                    <span className="text-xs text-blue-200 uppercase tracking-wide print:text-gray-500">Left</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Meal List Grid */}
            <div className="space-y-4">
              {DAILY_MEALS.map((meal) => (
                <div key={meal.id} className="group rounded-xl bg-white p-5 shadow-sm border border-gray-100 transition-all hover:shadow-md hover:border-teal-100 print:shadow-none print:border-gray-300 print:break-inside-avoid">
                  <div className="flex flex-col md:flex-row gap-6 items-center">
                    <div className="flex-shrink-0 w-full md:w-32 text-center md:text-left md:border-r md:border-gray-100 md:pr-6">
                      <span className="text-4xl block mb-2">{meal.img}</span>
                      <p className="text-xs font-bold text-gray-400 uppercase tracking-wider">{meal.type}</p>
                      <div className="flex items-center justify-center md:justify-start gap-1 text-xs text-gray-500 mt-1">
                        <Icons.Clock /> {meal.time}
                      </div>
                    </div>
                    <div className="flex-grow text-center md:text-left">
                      <h3 className="text-lg font-bold text-gray-900 group-hover:text-teal-600 transition-colors print:text-black">{meal.title}</h3>
                      <p className="text-sm text-gray-500 mt-1">{meal.calories} • {meal.protein}</p>
                    </div>
                    <div className="flex gap-3 w-full md:w-auto print:hidden">
                      <button onClick={() => setSelectedRecipe(meal)} className="flex-1 md:flex-none px-4 py-2 rounded-lg border border-gray-200 text-sm font-medium text-gray-700 hover:bg-gray-50 hover:border-gray-300 transition">View Recipe</button>
                      <button onClick={handleLogItem} className="flex-1 md:flex-none px-4 py-2 rounded-lg bg-teal-600 text-sm font-medium text-white hover:bg-teal-700 shadow-sm transition">Log Meal</button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </>
        )}

        {/* WEEKLY VIEW CONTENT */}
        {viewMode === "weekly" && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6 animate-in fade-in slide-in-from-bottom-2">
             {WEEKLY_PLAN.map((day, idx) => (
               <div key={idx} className="bg-white rounded-xl shadow-sm border border-gray-100 p-5 hover:shadow-md transition print:break-inside-avoid print:border-gray-300">
                  <h3 className="font-bold text-lg text-teal-700 mb-4 pb-2 border-b">{day.day}</h3>
                  <div className="space-y-4">
                    <div>
                      <span className="text-xs font-bold text-gray-400 uppercase">Breakfast</span>
                      <p className="text-sm font-medium text-gray-800">{day.b}</p>
                    </div>
                    <div>
                      <span className="text-xs font-bold text-gray-400 uppercase">Lunch</span>
                      <p className="text-sm font-medium text-gray-800">{day.l}</p>
                    </div>
                    <div>
                      <span className="text-xs font-bold text-gray-400 uppercase">Dinner</span>
                      <p className="text-sm font-medium text-gray-800">{day.d}</p>
                    </div>
                  </div>
               </div>
             ))}
             {/* Weekly Shopping List Card */}
             <div className="bg-teal-50 rounded-xl border border-teal-100 p-5 flex flex-col justify-center items-center text-center hover:shadow-md transition print:hidden">
                <div className="h-12 w-12 bg-teal-100 text-teal-600 rounded-full flex items-center justify-center mb-3">
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M3 3h2l.4 2M7 13h10l4-8H5.4M7 13L5.4 5M7 13l-2.293 2.293c-.63.63-.184 1.707.707 1.707H17m0 0a2 2 0 100 4 2 2 0 000-4zm-8 2a2 2 0 11-4 0 2 2 0 014 0z" /></svg>
                </div>
                <h3 className="font-bold text-gray-900">Shopping List</h3>
                <p className="text-xs text-gray-500 mt-1 mb-4">View ingredients for the week</p>
                <button className="px-4 py-2 bg-teal-600 text-white text-sm font-medium rounded-lg hover:bg-teal-700 transition">View List</button>
             </div>
          </div>
        )}

      </main>
    </div>
  );
}