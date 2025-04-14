// lib/api.ts

import axios from 'axios';

// Define the base URL for the API
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000';

// --- Interfaces ---
export interface Company {
  sector: string;
  company_name: string;
  fin_code: string;
  symbol?: string;
  options_available?: string[];
}

export interface SearchResponse {
  matches: {
    company_name: string;
    fin_code: string;
    score: number;
  }[];
}

export interface SessionResponse {
  session_id: string;
  message: string;
  company?: {
    company_name: string;
    industry: string;
    sector: string;
    fin_code:string;
  };
}

export interface QuestionResponse {
  answer: string;
  used: 'local_model' | 'condensed' | 'full_data';
  company_data_used?: any;
}



// Define interfaces for API responses
export interface Company {
  sector: string;
  company_name: string;
  fin_code: string;
  options_available?: string[];
}

export interface SearchResponse {
  companies: Company[];
  message?: string;
}

export interface SessionResponse {
  company_count: any;
  screener_name: any;
  session_id: string;
  message: string;
  company_name?: string;
  logo_url?: string;
}

export interface QuestionResponse {
  answer: string;
  status: string;
  question: string;
  llm_answer: string;
  model: string;
  is_first_question: boolean;
}

// API function to search for companies
export async function searchCompanies(query: string): Promise<SearchResponse> {
  try {
    const res = await axios.post(`${API_BASE_URL}/search-company`, { query });
    return res.data;
  } catch (error: any) {
    throw new Error(error.response?.data?.error || 'Failed to search for companies');
  }
}

 

export async function startSession(finCode: string): Promise<SessionResponse> {
  try {
    const res = await axios.post(`${API_BASE_URL}/start-session`, {
      fin_code: finCode // ✅ CORRECT KEY
    });
    return res.data;
  } catch (error: any) {
    throw new Error(error.response?.data?.error || 'Failed to start session');
  }
}

// API function to start a session for a screener
export async function startScreenerSession(screenerKeyword: string): Promise<SessionResponse> {
  try {
    const response = await axios.post(`${API_BASE_URL}/start-screener-session`, {
      screener_keyword: screenerKeyword
    });
    return response.data;
  } catch (error: any) {
    throw new Error(error.response?.data?.error || 'Failed to start screener session');
  }
}

// API function to ask a question within a session
// export async function askQuestion(sessionId: string, question: string): Promise<QuestionResponse> {
//   try {
//     const response = await axios.post(`${API_BASE_URL}/ask`, {
//       session_id: sessionId,
//       question: question
//     });
//     return response.data;
//   } catch (error: any) {
//     throw new Error(error.response?.data?.error || 'Failed to process question');
//   }
// }

// API function to ask a question about a screener
// export async function askScreenerQuestion(question: string): Promise<any> {
//   try {
//     const response = await axios.post(`${API_BASE_URL}/screener-question`, {
//       question: question
//     });
//     return response.data;
//   } catch (error: any) {
//     throw new Error(error.response?.data?.error || 'Failed to process screener question');
//   }
// }

// API function to clear/end a session
// export async function clearSession(sessionId: string): Promise<any> {
//   try {
//     const response = await axios.post(`${API_BASE_URL}/clear-session`, {
//       session_id: sessionId
//     });
//     return response.data;
//   } catch (error: any) {
//     console.error('Error clearing session:', error);
//     // Don't throw here, as this is typically called during cleanup
//     return { error: error.response?.data?.error || 'Failed to clear session' };
//   }
// }

export async function askQuestion(sessionId: string, question: string): Promise<QuestionResponse> {
  try {
    const res = await axios.post(`${API_BASE_URL}/company-question`, {
      session_id: sessionId,
      question
    });
    return res.data;
  } catch (error: any) {
    throw new Error(error.response?.data?.error || 'Failed to process question');
  }
}


export async function askScreenerQuestion(question: string): Promise<any> {
  try {
    const res = await axios.post(`${API_BASE_URL}/ask`, {
      question
    });
    return res.data;
  } catch (error: any) {
    throw new Error(error.response?.data?.error || 'Failed to process screener question');
  }
}

export async function clearSession(sessionId: string): Promise<any> {
  // You can implement this in Flask if needed
  return { status: 'not_implemented' };
}

// Translation (if implemented in backend — optional)
export async function translateToEnglish(text: string): Promise<{
  translatedText: string;
  detectedScript: string;
}> {
  // You can implement this in Flask if needed
  return {
    translatedText: text,
    detectedScript: 'latin'
  };
}

// API function to translate non-English queries to English
// export async function translateToEnglish(text: string): Promise<{ 
//   translatedText: string;
//   detectedScript: string;
// }> {
//   try {
//     const response = await axios.post(`${API_BASE_URL}/translate`, { text });
//     return response.data;
//   } catch (error: any) {
//     throw new Error(error.response?.data?.error || 'Failed to translate text');
//   }
// }