'use client';

import { useState } from 'react';
import { searchCompanies, translateToEnglish } from '@/lib/api';
import { Company } from '@/lib/api';

interface SearchBarProps {
  onSearchResults: (companies: Company[]) => void;
  onError: (message: string) => void;
  onLoading: (isLoading: boolean) => void;
}

const SearchBar = ({ onSearchResults, onError, onLoading }: SearchBarProps) => {
  const [query, setQuery] = useState('');
  const [isTranslating, setIsTranslating] = useState(false);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!query.trim()) {
      onError('Please enter a search term');
      return;
    }
    
    try {
      onLoading(true);
      setIsTranslating(false);
      
      // Check if query contains non-Latin characters (needs translation)
      const needsTranslation = /[^\u0000-\u007F]/.test(query);
      
      let searchQuery = query;
      
      // Translate if needed
      if (needsTranslation) {
        setIsTranslating(true);
        try {
          const translationResult = await translateToEnglish(query);
          searchQuery = translationResult.translatedText;
          
          console.log(
            `Translated: "${query}" → "${searchQuery}" (${translationResult.detectedScript})`
          );
        } catch (translationError) {
          console.warn('Translation failed, using original query:', translationError);
          // Continue with original query if translation fails
        } finally {
          setIsTranslating(false);
        }
      }
      
      // Search using the possibly translated query
      try {
        const response = await searchCompanies(searchQuery);
        onSearchResults(response.companies || []);
      } catch (searchError: any) {
        // If search with translated query fails, try with original query
        if (needsTranslation && searchQuery !== query) {
          console.warn('Search with translated query failed, trying original query');
          
          try {
            const fallbackResponse = await searchCompanies(query);
            onSearchResults(fallbackResponse.companies || []);
          } catch (fallbackError: any) {
            // Both searches failed
            if (fallbackError.response?.status === 404) {
              onError(`No companies found matching "${query}"`);
            } else {
              onError(`Error searching for "${query}": ${fallbackError.message || 'Server error'}`);
            }
          }
        } else {
          // Original search failed
          if (searchError.response?.status === 404) {
            onError(`No companies found matching "${query}"`);
          } else {
            onError(`Error searching for "${query}": ${searchError.message || 'Server error'}`);
          }
        }
      }
      
      onLoading(false);
    } catch (error: any) {
      onLoading(false);
      onError(`Error: ${error.message || 'Unknown error'}`);
    }
  };

  return (
    <div className="w-full max-w-md mx-auto">
      <form onSubmit={handleSearch} className="flex w-full items-center">
        <div className="relative w-full">
          <div className="absolute inset-y-0 left-0 flex items-center pl-3 pointer-events-none">
            <svg className="w-4 h-4 text-gray-500 dark:text-gray-400" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 20 20">
              <path stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="m19 19-4-4m0-7A7 7 0 1 1 1 8a7 7 0 0 1 14 0Z"/>
            </svg>
          </div>
          <input
            type="search"
            className="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-full pl-10 p-2.5"
            placeholder="Search for company in any language (e.g., Tata, टाटा, 腾讯)"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            required
          />
        </div>
        <button
          type="submit"
          className="p-2.5 ml-2 text-sm font-medium text-white bg-blue-700 rounded-lg border border-blue-700 hover:bg-blue-800 focus:ring-4 focus:outline-none focus:ring-blue-300"
          disabled={isTranslating}
        >
          {isTranslating ? (
            <div className="h-4 w-4 animate-spin rounded-full border-2 border-solid border-white border-r-transparent"></div>
          ) : (
            <span>Search</span>
          )}
        </button>
      </form>
      {isTranslating && (
        <p className="text-xs text-gray-500 mt-1">Translating your query...</p>
      )}
    </div>
  );
};

export default SearchBar;