const SCORE_API_BASE = import.meta.env.VITE_SCORE_API || (import.meta.env.DEV ? 'http://127.0.0.1:2220' : '/score-api')
export const SCORE_API = SCORE_API_BASE

// API will be proxied through Vite to Flask backend at port 2225
export const API_BASE = '/api'
