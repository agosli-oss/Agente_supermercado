# Imagen base de Python
FROM python:3.11-slim

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar dependencias e instalarlas
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código y el CSV
COPY agente.py .
COPY archivo.csv .

# Puerto que expone Gradio
EXPOSE 7860

# Comando para arrancar la app
CMD ["python", "agente.py"]
