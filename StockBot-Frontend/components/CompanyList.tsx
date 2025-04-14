'use client';

import { Company } from '@/lib/api';

interface CompanyListProps {
  companies: Company[];
  onSelectCompany: (company: Company) => void;
}

const CompanyList = ({ companies, onSelectCompany }: CompanyListProps) => {
  if (!companies || companies.length === 0) {
    return <div className="text-center py-4">No companies found</div>;
  }

  return (
    <div className="mt-4">
      <h2 className="text-xl font-semibold mb-3">Companies</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {companies.map((company, index) => (
          <div 
            key={`${company.fin_code}-${index}`}
            className="bg-white p-4 rounded-lg shadow hover:shadow-md transition-shadow border border-gray-200 cursor-pointer"
            onClick={() => onSelectCompany(company)}
          >
            <h3 className="font-medium text-lg text-blue-800">{company.company_name}</h3>
            <p className="text-sm text-gray-500 mt-1">ID: {company.fin_code}</p>
            <div className="mt-3 flex flex-wrap gap-2">
              {company.options_available && company.options_available.map((option, i) => (
                <span 
                  key={i} 
                  className="inline-flex items-center px-2 py-1 rounded text-xs font-medium bg-blue-100 text-blue-800"
                >
                  {option}
                </span>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default CompanyList;