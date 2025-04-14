/** @type {import('next').NextConfig} */
const nextConfig = {
  // Uncomment this if you need API routes proxying to avoid CORS issues
  /* 
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://127.0.0.1:5000/:path*',
      },
    ]
  },
  */
}

module.exports = nextConfig