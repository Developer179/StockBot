'use client';

import { useState } from 'react';

interface QuestionFormProps {
  onSubmit: (question: string) => void;
  isDisabled?: boolean;
  placeholder?: string;
}

const QuestionForm = ({ 
  onSubmit, 
  isDisabled = false,
  placeholder = "Type your question..."
}: QuestionFormProps) => {
  const [question, setQuestion] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!question.trim()) return;
    
    onSubmit(question);
    setQuestion('');
  };

  return (
    <form onSubmit={handleSubmit} className="flex items-end gap-2">
      <div className="flex-grow">
        <label htmlFor="question" className="block text-sm font-medium text-gray-700 mb-1">
          Ask a question
        </label>
        <input
          id="question"
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder={placeholder}
          disabled={isDisabled}
          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-blue-500 focus:border-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
        />
      </div>
      <button
        type="submit"
        disabled={isDisabled || !question.trim()}
        className="px-5 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 focus:ring-4 focus:ring-blue-300 disabled:opacity-50 disabled:cursor-not-allowed"
      >
        Send
      </button>
    </form>
  );
};

export default QuestionForm;