from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import json
import asyncio
import httpx
import os
from typing import List, Dict

app = FastAPI()

# Configurar CORS (permite peticiones desde cualquier origen)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Montar el directorio de archivos estáticos (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configurar el cliente de OpenAI para que use la API de MiniMax
api_key = os.getenv("MINIMAX_API_KEY")
client = OpenAI(
    base_url="https://api.minimax.io/v1",  # Cambiado de .com a .io
    api_key=api_key
)

# Definir una herramienta de ejemplo: consulta del clima
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Obtiene el clima actual de una ciudad. El usuario debe proporcionar el nombre de la ciudad.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "Nombre de la ciudad, por ejemplo: La Habana, Madrid, Ciudad de México"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

async def get_weather(location: str) -> str:
    """
    Llama a la API gratuita de clima wttr.in para obtener el clima actual.
    Parámetro:
        location: nombre de la ciudad
    Retorna:
        Cadena JSON con la información del clima
    """
    try:
        async with httpx.AsyncClient() as client_http:
            url = f"https://wttr.in/{location}?format=j1"
            response = await client_http.get(url, timeout=10.0)
            
            if response.status_code == 200:
                data = response.json()
                current = data['current_condition'][0]
                
                weather_info = {
                    "location": location,
                    "temperature": f"{current['temp_C']}°C",
                    "feels_like": f"{current['FeelsLikeC']}°C",
                    "condition": current['weatherDesc'][0]['value'],
                    "humidity": f"{current['humidity']}%",
                    "wind_speed": f"{current['windspeedKmph']} km/h",
                    "wind_direction": current['winddir16Point'],
                    "pressure": f"{current['pressure']} mb",
                    "visibility": f"{current['visibility']} km",
                    "uv_index": current['uvIndex']
                }
                
                return json.dumps(weather_info, ensure_ascii=False)
            else:
                return json.dumps({"error": f"No se pudo obtener el clima de {location}"}, ensure_ascii=False)
                
    except Exception as e:
        return json.dumps({"error": f"Error al consultar el clima: {str(e)}"}, ensure_ascii=False)

@app.get("/")
async def read_root():
    """Redirige a la página principal (interfaz de chat)"""
    from fastapi.responses import FileResponse
    return FileResponse("static/index.html")

@app.post("/api/chat/stream")
async def chat_stream(request: Request):
    """API de chat en streaming (soporta llamadas a herramientas como el clima)"""
    try:
        body = await request.json()
        messages = body.get("messages", [])
        
        if not messages:
            return {"error": "El campo 'messages' no puede estar vacío"}
        
        async def generate():
            try:
                # Primera llamada al modelo (con streaming para ver el razonamiento)
                stream = client.chat.completions.create(
                    model="MiniMax-M2",
                    messages=messages,
                    tools=tools,
                    extra_body={"reasoning_split": True},
                    stream=True,
                )
                
                full_content = ""
                tool_calls_list = []
                reasoning_content = []
                
                # Procesar la respuesta en streaming
                for chunk in stream:
                    try:
                        delta = chunk.choices[0].delta
                    except Exception:
                        continue
                    
                    # Enviar el razonamiento (thinking) en tiempo real
                    rd = getattr(delta, "reasoning_details", None)
                    if rd:
                        for detail in rd:
                            if isinstance(detail, dict) and "text" in detail and detail["text"]:
                                reasoning_content.append(detail["text"])
                                data = json.dumps({"type": "thinking", "content": detail["text"]}, ensure_ascii=False)
                                yield f"data: {data}\n\n"
                                await asyncio.sleep(0.01)
                    
                    # Acumular el contenido por si hay que usarlo después
                    content_fragment = getattr(delta, "content", None)
                    if content_fragment:
                        full_content += content_fragment
                    
                    # Acumular llamadas a herramientas (tool calls)
                    tool_calls = getattr(delta, "tool_calls", None)
                    if tool_calls:
                        for tc in tool_calls:
                            idx = tc.index if hasattr(tc, "index") else 0
                            while len(tool_calls_list) <= idx:
                                tool_calls_list.append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                            
                            if hasattr(tc, "id") and tc.id:
                                tool_calls_list[idx]["id"] = tc.id
                            if hasattr(tc, "function"):
                                if hasattr(tc.function, "name") and tc.function.name:
                                    tool_calls_list[idx]["function"]["name"] = tc.function.name
                                if hasattr(tc.function, "arguments") and tc.function.arguments:
                                    tool_calls_list[idx]["function"]["arguments"] += tc.function.arguments
                
                # Si hay herramientas que ejecutar...
                if tool_calls_list:
                    # Notificar al usuario que se está usando una herramienta
                    for tool_call in tool_calls_list:
                        tool_info = f"🔧 Ejecutando herramienta: {tool_call['function']['name']}\n"
                        data = json.dumps({"type": "content", "content": tool_info}, ensure_ascii=False)
                        yield f"data: {data}\n\n"
                        await asyncio.sleep(0.01)
                    
                    # Añadir la respuesta del asistente al historial
                    messages.append({
                        "role": "assistant",
                        "content": full_content or None,
                        "tool_calls": tool_calls_list
                    })
                    
                    # Ejecutar cada herramienta
                    for tool_call in tool_calls_list:
                        function_name = tool_call['function']['name']
                        function_args = json.loads(tool_call['function']['arguments'])
                        
                        if function_name == "get_weather":
                            location = function_args.get("location")
                            tool_result = await get_weather(location)
                            
                            # Mostrar que se obtuvo el resultado
                            result_info = f"📊 Clima obtenido para {location}\n"
                            data = json.dumps({"type": "content", "content": result_info}, ensure_ascii=False)
                            yield f"data: {data}\n\n"
                            await asyncio.sleep(0.01)
                            
                            # Añadir el resultado de la herramienta al historial
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call['id'],
                                "content": tool_result
                            })
                    
                    # Segunda llamada al modelo (con el resultado de la herramienta) para dar la respuesta final
                    stream = client.chat.completions.create(
                        model="MiniMax-M2",
                        messages=messages,
                        tools=tools,
                        extra_body={"reasoning_split": True},
                        stream=True,
                    )
                    
                    # Procesar la respuesta final en streaming
                    for chunk in stream:
                        try:
                            delta = chunk.choices[0].delta
                        except Exception:
                            continue
                        
                        # Razonamiento
                        rd = getattr(delta, "reasoning_details", None)
                        if rd:
                            for detail in rd:
                                if isinstance(detail, dict) and "text" in detail and detail["text"]:
                                    data = json.dumps({"type": "thinking", "content": detail["text"]}, ensure_ascii=False)
                                    yield f"data: {data}\n\n"
                                    await asyncio.sleep(0.01)
                        
                        # Contenido de la respuesta
                        content_fragment = getattr(delta, "content", None)
                        if content_fragment:
                            data = json.dumps({"type": "content", "content": content_fragment}, ensure_ascii=False)
                            yield f"data: {data}\n\n"
                            await asyncio.sleep(0.01)
                
                else:
                    # Sin herramientas: enviar el contenido acumulado en fragmentos
                    if full_content:
                        chunk_size = 10
                        for i in range(0, len(full_content), chunk_size):
                            chunk_text = full_content[i:i+chunk_size]
                            data = json.dumps({"type": "content", "content": chunk_text}, ensure_ascii=False)
                            yield f"data: {data}\n\n"
                            await asyncio.sleep(0.01)
                
                # Señal de finalización
                yield "data: [DONE]\n\n"
                
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                yield f"data: {error_msg}\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
        
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/health")
async def health_check():
    """Verifica que el servicio esté funcionando"""
    return {"status": "ok", "message": "Servicio de chat con MiniMax funcionando"}

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("Servicio de chat con MiniMax iniciado...")
    print("Accede en: http://localhost:8000")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)
