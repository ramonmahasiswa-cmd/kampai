"""
Kampus AI — Backend FastAPI
Mendukung 5 provider AI + Google Auth + Wikipedia/arXiv
"""
import os, base64, json
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import httpx
import urllib.parse

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = FastAPI(title="Kampus AI Backend")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

KEYS = {
    "gpt":      os.getenv("OPENAI_API_KEY", ""),
    "gemini":   os.getenv("GEMINI_API_KEY", ""),
    "grok":     os.getenv("XAI_API_KEY", ""),
    "deepseek": os.getenv("DEEPSEEK_API_KEY", ""),
    "claude":   os.getenv("ANTHROPIC_API_KEY", ""),
}

MODELS = {
    "gpt":      {"fast": "gpt-4o-mini", "strong": "gpt-4o"},
    "gemini":   {"fast": "gemini-1.5-flash", "strong": "gemini-1.5-pro"},
    "grok":     {"fast": "grok-beta", "strong": "grok-2-latest"},
    "deepseek": {"fast": "deepseek-chat", "strong": "deepseek-reasoner"},
    "claude":   {"fast": "claude-3-5-haiku-20241022", "strong": "claude-3-5-sonnet-20241022"},
}

ENDPOINTS = {
    "gpt":      "https://api.openai.com/v1/chat/completions",
    "gemini":   "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
    "grok":     "https://api.x.ai/v1/chat/completions",
    "deepseek": "https://api.deepseek.com/v1/chat/completions",
    "claude":   "https://api.anthropic.com/v1/messages",
}

class ChatRequest(BaseModel):
    provider: str
    model: str = "fast"
    system: str
    message: str
    history: Optional[List[dict]] = []
    image: Optional[str] = None

class ArticleRequest(BaseModel):
    query: str
    source: str = "wikipedia"

class GoogleAuthRequest(BaseModel):
    credential: str

@app.get("/")
def root():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"status": "ok"}

@app.get("/api/health")
def health():
    return {"status": "healthy", "providers": {k: bool(v) for k, v in KEYS.items()}}

@app.post("/api/chat")
async def chat(req: ChatRequest):
    p = req.provider.lower()
    if p not in KEYS: raise HTTPException(400, f"Provider {p} tidak dikenal")
    if not KEYS[p]: raise HTTPException(500, f"API key {p} belum di-set")
    model = MODELS[p].get(req.model, MODELS[p]["fast"])
    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            if p in ("gpt", "grok", "deepseek"):
                reply = await _call_openai(client, p, model, req)
            elif p == "claude":
                reply = await _call_anthropic(client, model, req)
            elif p == "gemini":
                reply = await _call_gemini(client, model, req)
        return {"reply": reply, "provider": p, "model": model}
    except httpx.TimeoutException:
        raise HTTPException(504, "AI timeout")
    except Exception as e:
        raise HTTPException(500, str(e))

async def _call_openai(client, provider, model, req):
    msgs = [{"role":"system","content":req.system}]
    for h in (req.history or [])[-10:]:
        msgs.append({"role": h.get("role","user"), "content": h.get("content","")})
    msgs.append({"role":"user","content":req.message})
    r = await client.post(ENDPOINTS[provider], json={
        "model":model, "messages":msgs, "temperature":0.7,
        "max_tokens":2000 if req.model=="strong" else 1000,
    }, headers={"Authorization":f"Bearer {KEYS[provider]}","Content-Type":"application/json"})
    if r.status_code != 200: raise HTTPException(r.status_code, r.text)
    return r.json()["choices"][0]["message"]["content"]

async def _call_anthropic(client, model, req):
    msgs = [{"role":"user" if h.get("role")=="user" else "assistant","content":h.get("content","")} for h in (req.history or [])[-10:]]
    msgs.append({"role":"user","content":req.message})
    r = await client.post(ENDPOINTS["claude"], json={
        "model":model, "system":req.system, "messages":msgs,
        "max_tokens":2000 if req.model=="strong" else 1000,
    }, headers={"x-api-key":KEYS["claude"],"anthropic-version":"2023-06-01","Content-Type":"application/json"})
    if r.status_code != 200: raise HTTPException(r.status_code, r.text)
    return r.json()["content"][0]["text"]

async def _call_gemini(client, model, req):
    contents = [{"role":"user" if h.get("role")=="user" else "model","parts":[{"text":h.get("content","")}]} for h in (req.history or [])[-10:]]
    contents.append({"role":"user","parts":[{"text":req.message}]})
    url = ENDPOINTS["gemini"].format(model=model) + f"?key={KEYS['gemini']}"
    r = await client.post(url, json={
        "systemInstruction":{"parts":[{"text":req.system}]},
        "contents":contents,
        "generationConfig":{"temperature":0.7,"maxOutputTokens":2000 if req.model=="strong" else 1000},
    })
    if r.status_code != 200: raise HTTPException(r.status_code, r.text)
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]

@app.post("/api/articles")
async def articles(req: ArticleRequest):
    async with httpx.AsyncClient(timeout=20.0) as client:
        if req.source == "wikipedia":
            return await _wiki(client, req.query)
        elif req.source == "arxiv":
            return await _arxiv(client, req.query)
        else:
            return {"source":"scholar","articles":[{"title":f"Google Scholar: {req.query}","extract":"Klik untuk membuka.","url":f"https://scholar.google.com/scholar?q={urllib.parse.quote(req.query)}","source":"Google Scholar"}]}

async def _wiki(client, query):
    r = await client.get("https://id.wikipedia.org/w/api.php", params={
        "action":"query","format":"json","list":"search","srsearch":query,"srlimit":10
    })
    data = r.json()
    results = data.get("query",{}).get("search",[])
    articles = []
    for item in results:
        title = item["title"]
        try:
            rs = await client.get(f"https://id.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title)}")
            if rs.status_code == 200:
                s = rs.json()
                articles.append({
                    "title": s.get("title", title),
                    "extract": s.get("extract",""),
                    "thumbnail": (s.get("thumbnail") or {}).get("source",""),
                    "url": (s.get("content_urls",{}).get("desktop",{}) or {}).get("page", f"https://id.wikipedia.org/wiki/{urllib.parse.quote(title)}"),
                    "source": "Wikipedia"
                })
        except: continue
    return {"source":"wikipedia","articles":articles}

async def _arxiv(client, query):
    import re
    r = await client.get("http://export.arxiv.org/api/query", params={"search_query":f"all:{query}","max_results":10})
    entries = re.findall(r"<entry>(.*?)</entry>", r.text, re.DOTALL)
    articles = []
    for e in entries:
        title = re.search(r"<title>(.*?)</title>", e, re.DOTALL)
        summary = re.search(r"<summary>(.*?)</summary>", e, re.DOTALL)
        link = re.search(r"<id>(.*?)</id>", e)
        articles.append({
            "title": (title.group(1).strip() if title else ""),
            "extract": (summary.group(1).strip()[:500] if summary else ""),
            "url": (link.group(1).strip() if link else ""),
            "source": "arXiv"
        })
    return {"source":"arxiv","articles":articles}

@app.post("/api/search")
async def search(req: ArticleRequest):
    async with httpx.AsyncClient(timeout=20.0) as client:
        result = await _wiki(client, req.query)
        result["google_url"] = f"https://www.google.com/search?q={urllib.parse.quote(req.query)}"
        return result

@app.post("/api/auth/google")
async def google_auth(req: GoogleAuthRequest):
    try:
        import base64
        parts = req.credential.split(".")
        payload_b64 = parts[1] + "=" * (-len(parts[1]) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
        return {"user": {
            "email": payload.get("email"),
            "name": payload.get("name"),
            "picture": payload.get("picture"),
            "sub": payload.get("sub")
        }}
    except Exception as e:
        raise HTTPException(500, f"Google auth error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

    