
'use client';

import { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { nanoid } from 'nanoid';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://34.66.22.225:5000/api/session';

// --- Interfaces ---
interface CompanyMatch {
  fin_code: string;
  name: string;
  symbol?: string;
  score?: number;
}

interface ChatMessage {
  id: string;
  isUser: boolean;
  text: string;
  company?: CompanyMatch;
  screenerKeyword?: string;
  companies?: any[];
  matches?: CompanyMatch[];
  error?: boolean;
  source?: string;
  ambiguityContextId?: string;
}

interface SmartAskResponse {
  type: string;
  answer?: string;
  company?: CompanyMatch;
  screener_keyword?: string;
  screener_results?: {
    keyword: string;
    title: string;
    description: string;
    total_companies: number;
    companies: any[];
  };
  matches?: CompanyMatch[];
  message?: string;
  ambiguity_context_id?: string;
  source?: string;
  processing_time_ms?: number;
  requested_criteria?: string[];
  companies_compared?: string[];
  companies_mentioned?: string[];
}

// --- Logger Utility ---
const logger = {
  info: console.log,
  warn: console.warn,
  error: console.error,
  debug: console.debug,
};

const ChatInterface = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: nanoid(),
      isUser: false,
      text: "Hello! I'm your stock market assistant. How can I help you with stocks or companies today?"
    },
  ]);
  const [inputValue, setInputValue] = useState('');
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const [isMobile, setIsMobile] = useState(false);

  // Check viewport width on mount and resize
  useEffect(() => {
    const checkViewportWidth = () => {
      setIsMobile(window.innerWidth < 768);
    };
    
    checkViewportWidth();
    window.addEventListener('resize', checkViewportWidth);
    
    return () => window.removeEventListener('resize', checkViewportWidth);
  }, []);

  useEffect(() => {
    // Scroll to bottom on new message or loading state change
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  // Focus input field after sending a message
  useEffect(() => {
    if (!loading && inputRef.current) {
      inputRef.current.focus();
    }
  }, [loading]);

  // --- Helper Functions ---
  const addMessage = (message: Omit<ChatMessage, 'id'>) => {
    setMessages(prev => [...prev, { ...message, id: nanoid() }]);
  };

  // Process backend response and add appropriate message(s)
  const handleBackendResponse = (backendResponse: SmartAskResponse) => {
    logger.info("Handling Backend Response:", backendResponse);

    const aiMessage: Omit<ChatMessage, 'id'> = {
      isUser: false,
      text: backendResponse.answer || backendResponse.message || "Sorry, I couldn't process that.",
      company: undefined,
      screenerKeyword: undefined,
      companies: undefined,
      matches: undefined,
      error: false,
      source: backendResponse.source,
      ambiguityContextId: undefined,
    };

    switch (backendResponse.type) {
      case 'company_response':
        aiMessage.company = backendResponse.company;
        break;
      case 'screener_response':
        aiMessage.screenerKeyword = backendResponse.screener_keyword;
        aiMessage.companies = backendResponse.screener_results?.companies;
        break;
      case 'ambiguous_company_clarification':
        aiMessage.text = backendResponse.message || "Please clarify which company:";
        aiMessage.matches = backendResponse.matches;
        aiMessage.ambiguityContextId = backendResponse.ambiguity_context_id;
        if (!aiMessage.ambiguityContextId) {
          logger.error("Ambiguity response received without context ID!");
          aiMessage.text = "Something went wrong getting clarification options. Please ask again.";
          aiMessage.error = true;
        }
        break;
      case 'error_understanding_request':
      case 'processing_error':
      case 'company_not_found':
      case 'data_fetch_failed':
      case 'screener_not_found':
      case 'error_context_lost':
        aiMessage.text = backendResponse.message || "Sorry, an error occurred.";
        aiMessage.error = true;
        break;
      case 'comparison_response':
      case 'general_finance_response':
      case 'clarification_needed':
      case 'off_topic':
        break;
      case 'multi_company_query':
        aiMessage.companies = backendResponse.companies_mentioned?.map(name => ({ name })) || [];
        break;
      default:
        aiMessage.text = backendResponse.answer || backendResponse.message || "Received an unexpected response type.";
        logger.warn("Unhandled backend response type:", backendResponse.type);
    }
    addMessage(aiMessage);
  };

  const renderCompanyList = (
    companies: CompanyMatch[] | undefined,
    title: string = "Please select a company:",
    onSelect: (contextId: string | undefined, name: string, finCode: string) => void,
    contextId: string | undefined,
    limit = 10
  ) => {
    if (!companies || companies.length === 0) return null;

    return (
      <div className="mt-2 text-sm text-gray-600">
        <p className="font-medium">{title}</p>
        <div className="flex flex-wrap gap-2 mt-2">
          {companies.slice(0, limit).map((company) => (
            company.fin_code ? (
              <button
                key={company.fin_code}
                className="bg-blue-50 text-blue-800 px-3 py-1.5 rounded-full hover:bg-blue-100 transition border border-blue-200 disabled:opacity-50 disabled:cursor-not-allowed"
                onClick={() => onSelect(contextId, company.name, company.fin_code)}
                disabled={loading}
              >
                {company.symbol || company.name}
              </button>
            ) : (
              <span key={company.name || nanoid()} className="text-red-500 text-xs italic">
                (Missing ID for {company.name})
              </span>
            )
          ))}
          {companies.length > limit && (
            <span className="px-3 py-1.5 rounded-full bg-gray-100 text-gray-600">
              +{companies.length - limit} more
            </span>
          )}
        </div>
      </div>
    );
  };

  // --- Handler for Ambiguity Resolution ---
  const handleCompanySelection = async (contextId: string | undefined, name: string, finCode: string) => {
    if (loading) return;
    if (!contextId) {
      logger.warn("Company selected without ambiguity context ID. Setting input field.");
      setInputValue(`Tell me more about ${name}`);
      return;
    }

    logger.info(`Resolving ambiguity with context: ${contextId}, selected: ${name} (${finCode})`);
    addMessage({
      isUser: false,
      text: `Okay, let's focus on ${name}. Processing your original question...`
    });
    setLoading(true);

    // Disable buttons for the original clarification message
    setMessages(prev => prev.map(msg =>
      msg.ambiguityContextId === contextId
        ? { ...msg, ambiguityContextId: undefined, matches: undefined }
        : msg
    ));

    try {
      const response = await axios.post<SmartAskResponse>(`${API_BASE_URL}/smart-ask`, {
        resolve_ambiguity_context_id: contextId,
        selected_fin_code: finCode
      });
      handleBackendResponse(response.data);
    } catch (error: any) {
      logger.error("Error resolving ambiguity:", error);
      let errorText = "Sorry, I couldn't apply your selection. Please try asking again.";
      if (axios.isAxiosError(error) && error.response) {
        const backendError = error.response.data;
        errorText = `Error: ${backendError?.message || error.response.statusText || 'Failed to resolve selection.'}`;
        if (backendError?.type === 'error_context_lost') {
          errorText = backendError.message || "Sorry, context lost. Please ask again with the company name.";
        }
      } else if (error.request) {
        errorText = "Error: Could not reach server to resolve selection.";
      }
      addMessage({ isUser: false, text: errorText, error: true });
    } finally {
      setLoading(false);
    }
  };

  // --- Main Submit Handler ---
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const userQuestion = inputValue.trim();
    if (!userQuestion || loading) return;

    setInputValue('');
    addMessage({ isUser: true, text: userQuestion });
    setLoading(true);

    try {
      const response = await axios.post<SmartAskResponse>(`${API_BASE_URL}/smart-ask`, {
        question: userQuestion.toUpperCase()
      });
      handleBackendResponse(response.data);
    } catch (error: any) {
      logger.error("Error calling /smart-ask:", error);
      let errorText = "Sorry, an error occurred connecting to the assistant.";
      if (axios.isAxiosError(error) && error.response) {
        errorText = `Error: ${error.response.data?.message || error.response.statusText || 'Server error.'}`;
      } else if (error.request) {
        errorText = "Error: Could not reach the server.";
      }
      addMessage({ isUser: false, text: errorText, error: true });
    } finally {
      setLoading(false);
    }
  };

  // --- JSX ---
  return (
    <div className="flex flex-col h-screen max-w-5xl mx-auto bg-white shadow-xl border border-gray-200 overflow-hidden">
      {/* Header */}
      <header className="p-3 md:p-4 bg-gradient-to-r from-blue-700 to-indigo-800 text-white shadow-md sticky top-0 z-10">
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <div className="bg-white bg-opacity-20 p-2 rounded-lg mr-3">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
            </div>
            <div>
              <h1 className="text-lg md:text-xl font-bold">Stock Market Assistant</h1>
              <p className="text-xs md:text-sm opacity-90">Powered by Univest Stock Advisory AI</p>
            </div>
          </div>
          <div className="hidden md:block">
            <span className="bg-green-500 px-2 py-1 rounded-full text-xs font-medium">ONLINE</span>
          </div>
        </div>
      </header>

      {/* Chat Area */}
      <main className="flex-grow overflow-y-auto px-3 md:px-6 py-4 bg-gray-50">
        <div className="space-y-4 max-w-3xl mx-auto">
          {messages.map((msg) => (
            <div
              key={msg.id}
              className={`flex ${msg.isUser ? 'justify-end' : 'justify-start'} animate-fadeIn`}
            >
              {/* AI Avatar */}
              {!msg.isUser && !isMobile && (
                <div className="w-8 h-8  flex items-center justify-center text-indigo-600 mr-2 flex-shrink-0 ">
                  {/* <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-6-3a2 2 0 11-4 0 2 2 0 014 0zm-2 4a5 5 0 00-4.546 2.916A5.986 5.986 0 0010 16a5.986 5.986 0 004.546-2.084A5 5 0 0012 11z" clipRule="evenodd" />
                  </svg> */}
                  <img src="/bot.jpg" className="rounded-full"  alt="" />
                </div>
              )}
              
              {/* Message Bubble */}
              <div
                className={`relative max-w-[85%] md:max-w-[75%] rounded-2xl px-4 py-3 shadow-sm 
                  ${msg.isUser ? 
                    'bg-blue-600 text-white rounded-br-none' : 
                    `${msg.error ? 'border-red-300 bg-red-50 text-red-700' : 'bg-white border border-gray-200 text-gray-800'} rounded-bl-none`
                  }`}
              >
                {/* Context Headers */}
                {msg.company && !msg.isUser && (
                  <div className="text-xs font-semibold mb-2 text-blue-700 border-b border-blue-100 pb-1 flex items-center">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-3 w-3 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 13.255A23.931 23.931 0 0112 15c-3.183 0-6.22-.62-9-1.745M16 6V4a2 2 0 00-2-2h-4a2 2 0 00-2 2v2m4 6h.01M5 20h14a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                    </svg>
                    {msg.company.name} {msg.company.symbol ? `(${msg.company.symbol})` : ''}
                  </div>
                )}
                
                {msg.screenerKeyword && !msg.isUser && (
                  <div className="text-xs font-semibold mb-2 text-indigo-700 border-b border-indigo-100 pb-1 flex items-center">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-3 w-3 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                    </svg>
                    {msg.screenerKeyword.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                  </div>
                )}

                {/* Main Text */}
                <p className="whitespace-pre-wrap text-sm md:text-base leading-relaxed">{msg.text}</p>

                {/* Render Buttons for Ambiguity */}
                {msg.matches && msg.matches.length > 0 && !msg.isUser && msg.ambiguityContextId &&
                  renderCompanyList(
                    msg.matches,
                    msg.text.includes("clarify") ? "Please select which company you meant:" : "Which company?",
                    handleCompanySelection,
                    msg.ambiguityContextId
                  )
                }

                {/* Render Example Companies for Screener */}
                {msg.companies && msg.companies.length > 0 && msg.screenerKeyword && !msg.isUser &&
                  renderCompanyList(
                    msg.companies,
                    "Example companies found:",
                    (ctxId, name, finCode) => { setInputValue(`Tell me more about ${name}`); },
                    undefined,
                    5
                  )
                }
                
                {/* Message timestamp - subtle and right-aligned */}
               {/*  <div className={`text-[10px] mt-1 ${msg.isUser ? 'text-blue-200 text-right' : 'text-gray-400 text-left'}`}>
                  {new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}
                </div> */}
              </div>
              
              {/* User Avatar */}
              {msg.isUser && !isMobile && (
                <div className="w-8 h-8 rounded-full bg-gray-300 flex items-center justify-center text-gray-600 ml-2 flex-shrink-0 shadow-sm border border-gray-400">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M10 9a3 3 0 100-6 3 3 0 000 6zm-7 9a7 7 0 1114 0H3z" clipRule="evenodd" />
                  </svg>
                </div>
              )}
            </div>
          ))}
          
          {/* Loading Indicator */}
          {loading && (
            <div className="flex justify-start">
              {/* AI Avatar */}
              {!isMobile && (
                <div className="w-8 h-8 rounded-full bg-indigo-100 flex items-center justify-center text-indigo-600 mr-2 flex-shrink-0 shadow-sm border border-indigo-200">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-6-3a2 2 0 11-4 0 2 2 0 014 0zm-2 4a5 5 0 00-4.546 2.916A5.986 5.986 0 0010 16a5.986 5.986 0 004.546-2.084A5 5 0 0012 11z" clipRule="evenodd" />
                  </svg>
                </div>
              )}
              
              {/* Bubble */}
              <div className="bg-white border border-gray-200 text-gray-800 rounded-2xl px-4 py-3 max-w-[75%] shadow-sm rounded-bl-none">
                {/* Dots */}
                <div className="flex space-x-2 items-center h-5">
                  <span className="inline-block w-2.5 h-2.5 bg-indigo-600 rounded-full animate-bounce delay-75"></span>
                  <span className="inline-block w-2.5 h-2.5 bg-indigo-600 rounded-full animate-bounce delay-150"></span>
                  <span className="inline-block w-2.5 h-2.5 bg-indigo-600 rounded-full animate-bounce delay-300"></span>
                </div>
              </div>
            </div>
          )}
          
          {/* Scroll anchor */}
          <div ref={messagesEndRef} />
        </div>
      </main>

      {/* Input Area */}
      <footer className="p-3 md:p-4 border-t bg-white shadow-inner">
        {/* Input Form */}
        <form onSubmit={handleSubmit} className="flex gap-2 items-center max-w-3xl mx-auto">
          <input 
            ref={inputRef}
            type="text" 
            value={inputValue} 
            onChange={(e) => setInputValue(e.target.value)} 
            placeholder="Ask about stocks..." 
            disabled={loading} 
            className="flex-grow px-4 py-3 border border-gray-300 rounded-full focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:opacity-60 transition duration-150 ease-in-out shadow-sm"
            autoFocus 
          />
          <button 
            type="submit" 
            disabled={loading || !inputValue.trim()} 
            className="p-3 w-12 h-12 bg-blue-600 text-white rounded-full hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition duration-150 ease-in-out flex items-center justify-center shadow-sm"
            aria-label="Send message"
          >
            {loading ? (
              <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" className="opacity-25"></circle>
                <path fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" className="opacity-75"></path>
              </svg>
            ) : (
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                <path d="M10.894 2.553a1 1 0 00-1.789 0l-7 14a1 1 0 001.169 1.409l5-1.429A1 1 0 009 15.571V11a1 1 0 112 0v4.571a1 1 0 00.725.962l5 1.428a1 1 0 001.17-1.408l-7-14z" />
              </svg>
            )}
          </button>
        </form>

        {/* Example Prompts */}
        {!loading && (
          <div className="mt-3 max-w-3xl mx-auto">
            <p className="text-xs text-gray-500 mb-1.5">Try asking:</p>
            <div className="flex flex-wrap gap-2">
              <button 
                onClick={() => setInputValue("What is the price of Tata Communications?")} 
                className="text-xs md:text-sm px-3 py-1.5 bg-white rounded-full hover:bg-gray-50 active:bg-gray-100 border border-gray-300 shadow-sm transition"
              >
                Price of Tata Communications?
              </button>
              <button 
                onClick={() => setInputValue("Compare Infosys and TCS")} 
                className="text-xs md:text-sm px-3 py-1.5 bg-white rounded-full hover:bg-gray-50 active:bg-gray-100 border border-gray-300 shadow-sm transition"
              >
                Compare Infosys and TCS
              </button>
              <button 
                onClick={() => setInputValue("Show me high growth IT stocks")} 
                className="text-xs md:text-sm px-3 py-1.5 bg-white rounded-full hover:bg-gray-50 active:bg-gray-100 border border-gray-300 shadow-sm transition"
              >
                High growth IT stocks
              </button>
              <button 
                onClick={() => setInputValue("Tell me about Tata")} 
                className="text-xs md:text-sm px-3 py-1.5 bg-white rounded-full hover:bg-gray-50 active:bg-gray-100 border border-gray-300 shadow-sm transition"
              >
                Tell me about Tata
              </button>
            </div>
          </div>
        )}
      </footer>
      
      {/* Add styles for fadeIn animation */}
      <style jsx global>{`
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .animate-fadeIn {
          animation: fadeIn 0.3s ease-out forwards;
        }
      `}</style>
    </div>
  );
};

export default ChatInterface;