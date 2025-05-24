import React from 'react';
import { Routes, Route } from 'react-router-dom';
import TranslationDemo from './components/TranslationDemo';
import './App.css';

const App: React.FC = () => {
  return (
    <div className="app">
      <header className="app-header">
        <h1>InfiniSST Demo</h1>
      </header>
      <main>
        <Routes>
          <Route path="/" element={<TranslationDemo />} />
        </Routes>
      </main>
    </div>
  );
};

export default App; 