'use client';

import { useState, useEffect } from 'react';
import { Company, startSession, askQuestion, clearSession } from '@/lib/api';
import QuestionForm from './QuestionForm';
import Image from 'next/image';

interface CompanyDetailsProps {
  company: Company;
  onBack: () => void;
}

interface response {
  logo_url: String;
}

interface ChatMessage {
  isUser: boolean;
  text: string;
}

const CompanyDetails = ({ company, onBack }: CompanyDetailsProps) => {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [logoUrl, setLogoUrl] = useState<string | null>(null);
  const [sessionActive, setSessionActive] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  
  // Start a session when the component mounts
  useEffect(() => {
    const initSession = async () => {
      try {
        setLoading(true);
        const response = await startSession(company.fin_code);
        setSessionId(response.session_id);
        setSessionActive(true);
        // TypeScript safe way to set logo URL
        if (typeof response.logo_url === 'string') {
          setLogoUrl(response.logo_url);
        } else {
          setLogoUrl(null);
        }
        
        // Add welcome message
        setChatMessages([
          {
            isUser: false,
            text: `I'm ready to answer questions about ${company.company_name}. What would you like to know?`
          }
        ]);
        
        setLoading(false);
      } catch (error: any) {
        setLoading(false);
        setError(`Failed to start session: ${error.message || 'Unknown error'}`);
      }
    };
    
    initSession();
    
    // Clean up session when component unmounts
    return () => {
      if (sessionId) {
        clearSession(sessionId).catch(console.error);
      }
    };
  }, [company]);
  
  const handleAskQuestion = async (question: string) => {
    if (!sessionId || !sessionActive) {
      setError('Session is not active. Please try again.');
      return;
    }
    
    // Add user question to chat
    setChatMessages(prev => [...prev, { isUser: true, text: question }]);
    
    try {
      setLoading(true);
      
      const response = await askQuestion(sessionId, question);
      
      // Add AI response to chat
      setChatMessages(prev => [...prev, { isUser: false, text: response.llm_answer }]);
      
      setLoading(false);
    } catch (error: any) {
      setLoading(false);
      
      // Add error message to chat
      setChatMessages(prev => [...prev, { 
        isUser: false, 
        text: `Sorry, I couldn't process your question. Error: ${error.message || 'Unknown error'}`
      }]);
      
      // Check if session expired
      if (error.response?.status === 404 && error.response?.data?.error?.includes('expired')) {
        setSessionActive(false);
        setError('Your session has expired. Please go back and select the company again.');
      }
    }
  };

  return (
    <div className="bg-white rounded-lg shadow p-6">
      {/* Enhanced header with logo and active status badge */}
      <div className="flex justify-between items-center mb-6 border-b pb-4">
        <div className="flex items-center">
          {/* Logo container with status indicator in bottom right */}
          <div className="relative mr-4">
            {logoUrl ? (
              <div className="w-12 h-12 rounded-md overflow-hidden flex-shrink-0 bg-gray-50 border border-gray-200">
                <img 
                  src={logoUrl} 
                  alt={`${company.company_name} logo`} 
                  className="w-full h-full object-contain"
                  onError={(e) => {
                    e.currentTarget.src = '/company-placeholder.svg';
                  }}
                />
              </div>
            ) : (
              <div className="w-12 h-12 rounded-md flex items-center justify-center bg-gray-100 text-gray-500 flex-shrink-0">
                {company.company_name.charAt(0)}
              </div>
            )}
            {/* Status indicator badge positioned in bottom right of logo */}
            {/* <div className="absolute bottom-0 right-0  w-3 h-3 rounded-full border-2 border-white shadow-sm" 
                 style={{ backgroundColor: sessionActive ? '#10B981' : '#EF4444' }}>
            </div> */}
          </div>
          <div>
            <div className="flex items-center">
              <h2 className="text-2xl font-bold text-gray-800">{company.company_name}</h2>
              <span className={`ml-2 px-2 py-0.5 text-xs font-medium rounded-full ${
                sessionActive ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
                {sessionActive ? 'Active' : 'Inactive'}
              </span>
            </div>
            <p className="text-sm text-gray-500 flex items-center">
              {company.sector || 'Financial Services'}
            </p>
          </div>
        </div>
        <button
          onClick={onBack}
          className="px-4 py-2 text-sm font-medium text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded-md transition-colors"
        >
          ← Back to results
        </button>
      </div>
      
      {error && (
        <div className="bg-red-100 border-l-4 border-red-500 text-red-700 p-4 mb-4 rounded-r-md">
          <p>{error}</p>
        </div>
      )}
      
      {/* Chat container */}
      <div className="bg-gray-50 rounded-lg p-4 mb-6 overflow-y-auto shadow-inner" style={{ height: '450px' }}>
        {chatMessages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full">
            {logoUrl && (
              <div className="w-20 h-20 mb-4 opacity-30">
                <img
                  src={logoUrl}
                  alt={`${company.company_name} logo`}
                  className="w-full h-full object-contain"
                  onError={(e) => {
                    e.currentTarget.src = '/company-placeholder.svg';
                  }}
                />
              </div>
            )}
            <p className="text-gray-500">
              {loading ? 'Initializing session...' : 'Ask a question about the company'}
            </p>
          </div>
        ) : (
          <div className="space-y-4">
            {chatMessages.map((msg, index) => (
              <div
                key={index}
                className={`flex ${msg.isUser ? 'justify-end' : 'justify-start'}`}
              >
                {!msg.isUser && (
                  <div className="w-8 h-8 mr-2 flex-shrink-0">
                    <img
                      src={logoUrl || '/company-placeholder.svg'}
                      alt="Company"
                      className="w-full h-full object-contain rounded-full border border-gray-200"
                      onError={(e) => {
                        e.currentTarget.src = '/company-placeholder.svg';
                      }}
                    />
                  </div>
                )}
                <div
                  className={`max-w-[75%] rounded-lg px-4 py-2 ${
                    msg.isUser 
                      ? 'bg-blue-600 text-white rounded-br-none' 
                      : 'bg-gray-200 text-gray-800 rounded-bl-none'
                  }`}
                >
                  <p className="whitespace-pre-wrap">{msg.text}</p>
                </div>
              </div>
            ))}
            {loading && (
              <div className="flex justify-start">
                <div className="w-8 h-8 mr-2 flex-shrink-0">
                  <img
                    src={logoUrl || '/company-placeholder.svg'}
                    alt="Company"
                    className="w-full h-full object-contain rounded-full border border-gray-200"
                    onError={(e) => {
                      e.currentTarget.src = '/company-placeholder.svg';
                    }}
                  />
                </div>
                <div className="bg-gray-200 text-gray-800 rounded-lg rounded-bl-none max-w-[75%] px-4 py-2">
                  <div className="flex gap-1">
                    <span className="animate-bounce">•</span>
                    <span className="animate-bounce delay-100">•</span>
                    <span className="animate-bounce delay-200">•</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
      
      <QuestionForm 
        onSubmit={handleAskQuestion} 
        isDisabled={loading || !sessionActive}
        placeholder="Ask about financials, stock performance, or company details..."
      />
      
      {/* Enhanced session info footer */}
      <div className="mt-4 text-xs text-gray-500 flex justify-between items-center p-2 bg-gray-50 rounded-md">
        <div className="flex items-center">
          <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <span>Session: {sessionId ? `${sessionId.substring(0, 8)}...` : 'Initializing...'}</span>
        </div>
        <div className="flex items-center">
          <div className="w-2 h-2 rounded-full mr-2" style={{ backgroundColor: sessionActive ? '#10B981' : '#EF4444' }}></div>
          <span className={sessionActive ? 'text-green-700' : 'text-red-700'}>
            {sessionActive ? 'Active' : 'Inactive'}
          </span>
        </div>
      </div>
    </div>
  );
};

export default CompanyDetails;