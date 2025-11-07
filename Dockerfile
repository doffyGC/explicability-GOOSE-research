# Imagem base leve e moderna
FROM python:3.11-slim

# Definir diretório de trabalho
WORKDIR /app

# Copiar os arquivos de dependência
COPY requirements.txt .

# Instalar dependências do sistema necessárias para compilar pacotes
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && pip install --upgrade pip \
    && pip install -r requirements.txt \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copiar o código-fonte do projeto
COPY . .

# Comando padrão (pode ser alterado)
CMD ["python", "main.py"]