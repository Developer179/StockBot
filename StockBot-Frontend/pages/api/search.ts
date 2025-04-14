import type { NextApiRequest, NextApiResponse } from 'next';
import axios from 'axios';

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse
) {
  try {
    const query = req.query.q;
    const response = await axios.get(`http://34.66.22.225:5000/search?q=${query}`);//TODO:check
    res.status(200).json(response.data);
  } catch (error) {
    console.error('Error proxying request:', error);
    res.status(500).json({ error: 'Failed to fetch data' });
  }
}