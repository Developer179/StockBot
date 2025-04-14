// 'use client';

// import { useState } from 'react';
// import SearchBar from '@/components/SearchBar';
// import CompanyList from '@/components/CompanyList';
// import CompanyDetails from '@/components/CompanyDetails';
// import { Company } from '@/lib/api';

// export default function Home() {
//   const [searchResults, setSearchResults] = useState<Company[]>([]);
//   const [selectedCompany, setSelectedCompany] = useState<Company | null>(null);
//   const [loading, setLoading] = useState(false);
//   const [error, setError] = useState<string | null>(null);

//   const handleSearchResults = (companies: Company[]) => {
//     setSearchResults(companies);
//     setError(null);
//     setSelectedCompany(null);
//   };

//   const handleError = (message: string) => {
//     setError(message);
//     setSearchResults([]);
//   };

//   const handleSelectCompany = (company: Company) => {
//     setSelectedCompany(company);
//   };

//   const handleBackToResults = () => {
//     setSelectedCompany(null);
//   };

//   return (
//     <main className="min-h-screen bg-gray-50 py-8">
//       <div className="container mx-auto px-4">
//         <header className="mb-8 text-center">
//           <h1 className="text-3xl font-bold text-gray-900 mb-2">Stock Market Assistant</h1>
//           <p className="text-gray-600">Search for companies and get AI-powered insights</p>
//         </header>

//         {!selectedCompany && (
//           <>
//             <SearchBar 
//               onSearchResults={handleSearchResults}
//               onError={handleError}
//               onLoading={setLoading}
//             />
            
//             {loading && (
//               <div className="text-center py-8">
//                 <div className="inline-block h-8 w-8 animate-spin rounded-full border-4 border-solid border-blue-600 border-r-transparent"></div>
//                 <p className="mt-2 text-gray-600">Searching for companies...</p>
//               </div>
//             )}
            
//             {error && (
//               <div className="mt-4 bg-red-100 border-l-4 border-red-500 text-red-700 p-4">
//                 <p>{error}</p>
//               </div>
//             )}
            
//             {!loading && !error && searchResults.length > 0 && (
//               <CompanyList 
//                 companies={searchResults}
//                 onSelectCompany={handleSelectCompany}
//               />
//             )}
//           </>
//         )}

//         {selectedCompany && (
//           <CompanyDetails 
//             company={selectedCompany}
//             onBack={handleBackToResults}
//           />
//         )}
//       </div>
//     </main>
//   );
// }


'use client';

import ChatInterface from '@/components/ChatInterFace';

export default function Home() {
  return (
    <main className="min-h-screen bg-gray-50 py-8">
      <div className="container mx-auto px-4">
        <header className="mb-8 text-center">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Stock Market Assistant</h1>
          <p className="text-gray-600">Ask questions about any company to get AI-powered insights</p>
        </header>

        <ChatInterface />
        
        <footer className="mt-8 text-center text-sm text-gray-500">
          <p>Powered by AI - Ask questions like "What's the financial outlook for Tata?" or "Tell me about Apple's recent performance"</p>
        </footer>
      </div>
    </main>
  );
}