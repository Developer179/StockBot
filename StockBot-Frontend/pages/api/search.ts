import type { NextApiRequest, NextApiResponse } from 'next';
import axios from 'axios';

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse
) {
  try {
    const query = req.query.q;
    const response = await axios.get(`http://127.0.0.1:5000/search?q=${query}`);
    res.status(200).json(response.data);
  } catch (error) {
    console.error('Error proxying request:', error);
    res.status(500).json({ error: 'Failed to fetch data' });
  }
}