import React, { useState } from "react";
import { Link } from "react-router-dom";

// --- Mock Data (LLM Placeholder) ---
const WEEKLY_WORKOUTS = [
  {
    id: 1,
    day: "Monday",
    type: "Cardio",
    duration: "30 min",
    intensity: "Moderate",
    activities: [
      { phase: "Warmup (5 min)", detail: "Light jogging in place, arm circles, leg swings." },
      { phase: "Main (25 min)", detail: "Treadmill or outdoor jogging - maintain steady heart rate (Zone 2)." },
      { phase: "Cooldown (5 min)", detail: "Slow walking & static stretching (hamstrings, quads)." }
    ]
  },
  {
    id: 2,
    day: "Wednesday",
    type: "Strength",
    duration: "35 min",
    intensity: "High",
    activities: [
      { phase: "Warmup (5 min)", detail: "Jumping jacks, bodyweight squats, push-ups (knees)." },
      { phase: "Main (25 min)", detail: "Circuit: 3 rounds of Dumbbell Squats, Overhead Press, and Lunges." },
      { phase: "Cooldown (5 min)", detail: "Child's pose, cat-cow stretch." }
    ]
  },
  {
    id: 3,
    day: "Friday",
    type: "Flexibility",
    duration: "20 min",
    intensity: "Low",
    activities: [
      { phase: "Warmup (5 min)", detail: "Neck rolls, shoulder shrugs." },
      { phase: "Main (15 min)", detail: "Full body yoga flow: Sun Salutation A & B." },
      { phase: "Cooldown", detail: "Deep breathing meditation." }
    ]
  }
];

// --- Sub-Components ---

// 1. Top Right Notification
const Toast = ({ show }) => (
  <div 
    className={`fixed top-5 right-5 z-50 transform rounded-md bg-emerald-500 px-6 py-3 text-white shadow-lg transition-all duration-300 ${
      show ? "translate-y-0 opacity-100" : "-translate-y-10 opacity-0 pointer-events-none"
    }`}
  >
    💪 Workout logged +10 points!
  </div>
);

// 2. Workout Details Modal
const WorkoutModal = ({ workout, onClose }) => {
  if (!workout) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/20 backdrop-blur-sm p-4">
      <div className="w-full max-w-md rounded-xl bg-white p-8 shadow-2xl animate-in fade-in zoom-in duration-200">
        <h2 className="text-xl font-bold text-gray-900">{workout.type} session - {workout.day}</h2>
        <p className="mt-1 text-sm text-gray-500">
          Duration: {workout.duration} • Intensity: {workout.intensity}
        </p>

        <div className="mt-6 space-y-6">
          {workout.activities.map((act, index) => (
            <div key={index}>
              <h3 className="font-semibold text-gray-900 text-sm">{act.phase}</h3>
              <ul className="mt-1 list-disc list-inside text-sm text-gray-600">
                <li>{act.detail}</li>
              </ul>
            </div>
          ))}
        </div>

        <button 
          onClick={onClose}
          className="mt-8 w-full rounded-md border border-gray-300 bg-white py-2 text-sm font-medium text-gray-700 hover:bg-gray-50"
        >
          Close
        </button>
      </div>
    </div>
  );
};

// --- Main Page Component ---
export default function ExercisePlan() {
  const [selectedWorkout, setSelectedWorkout] = useState(null);
  const [showToast, setShowToast] = useState(false);

  const handleLogItem = () => {
    setShowToast(true);
    setTimeout(() => setShowToast(false), 3000);
  };

  return (
    <main className="min-h-screen bg-gray-50 px-6 py-10 font-sans relative">
      
      <Toast show={showToast} />
      <WorkoutModal workout={selectedWorkout} onClose={() => setSelectedWorkout(null)} />

      <div className="mx-auto max-w-4xl">
        
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900">Your Exercise Plan</h1>
          <p className="mt-1 text-sm text-gray-500">30 - 45 mins/day | 3x per week</p>
        </div>

        {/* Blue Info Banner */}
        <div className="mb-8 rounded-md bg-blue-600 px-6 py-3 text-sm font-medium text-white shadow-sm">
          Your available time: 30 mins/day | Fitness level: Beginner | Goal: Lose weight, build strength
        </div>

        {/* Workout Cards List */}
        <div className="space-y-6">
          {WEEKLY_WORKOUTS.map((workout) => (
            <div key={workout.id} className="rounded-xl bg-gray-200 p-6 shadow-sm border border-gray-300">
              
              <h3 className="text-lg font-bold text-gray-900 mb-4">
                {workout.day} - {workout.type}
              </h3>

              {/* Steps Preview (White Bars) */}
              <div className="space-y-2 mb-6">
                {workout.activities.map((act, idx) => (
                  <div key={idx} className="rounded-md bg-white px-4 py-2 text-sm text-gray-700 shadow-sm">
                    <span className="font-semibold">{act.phase.split(' ')[0]}:</span> {act.detail}
                  </div>
                ))}
              </div>

              {/* Buttons */}
              <div className="flex justify-end gap-3">
                <button 
                  onClick={() => setSelectedWorkout(workout)}
                  className="rounded-md border border-teal-600 px-4 py-2 text-sm font-medium text-teal-700 bg-white transition hover:bg-teal-50"
                >
                  Watch Video
                </button>
                <button 
                  onClick={handleLogItem}
                  className="rounded-md bg-teal-700 px-4 py-2 text-sm font-medium text-white transition hover:bg-teal-800"
                >
                  Log Done
                </button>
              </div>

            </div>
          ))}
        </div>

        {/* Bottom Actions */}
        <div className="mt-10 flex flex-wrap gap-4">
          <button className="rounded-md bg-white border border-gray-300 px-6 py-2 text-sm font-medium text-gray-700 shadow-sm hover:bg-gray-50">
             Edit Plan
          </button>
          <button className="rounded-md bg-teal-700 px-6 py-2 text-sm font-medium text-white shadow-sm hover:bg-teal-800">
             Get Harder Plan
          </button>
          <button className="rounded-md bg-white border border-gray-300 px-6 py-2 text-sm font-medium text-gray-700 shadow-sm hover:bg-gray-50">
             Share Plan
          </button>
        </div>
        
        {/* Back Link */}
        <div className="mt-8">
           <Link to="/dashboard" className="text-sm text-gray-500 hover:underline">← Back to Dashboard</Link>
        </div>

      </div>
    </main>
  );
}