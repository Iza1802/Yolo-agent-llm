# AgroVision AI — Versão Completa
# YOLO + Ollama + Agente + Chat + Câmera Pública

## Pré-requisitos
- Python 3.11.15 (64 bits)
- VS Code com extensões Python e Pylance
- Ollama instalado (https://ollama.com/download)

---

## 1. Criar e ativar ambiente virtual

```powershell
cd C:\projetos\agrovision_ia
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Se o PowerShell bloquear:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

---

## 2. Instalar dependências

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Instalar e configurar o Ollama

### Baixar o modelo:
```powershell
ollama pull llama3
```

### Se a máquina for mais fraca, use o modelo menor:
```powershell
ollama pull llama3.2:3b
```
E no `.env`, altere: `OLLAMA_MODEL=llama3.2:3b`

### Subir o Ollama (em um terminal separado):
```powershell
ollama serve
```
Deixe esse terminal aberto.

---

## 4. Verificar o .env

Abra o arquivo `.env` e confirme:
- `OLLAMA_MODEL` bate com o modelo que você baixou
- `CAMERA_SOURCE` aponta para a câmera desejada

Câmera padrão (stream público Caltrans):
```
CAMERA_SOURCE=https://wzmedia.dot.ca.gov/D11/C214_SB_5_at_Via_De_San_Ysidro.stream/playlist.m3u8
```

Para usar webcam local, troque por:
```
CAMERA_SOURCE=0
```

---

## 5. Rodar o projeto

```powershell
python -m uvicorn app:app --reload
```

---

## 6. Abrir no navegador

- Dashboard: http://127.0.0.1:8000
- Status sistema: http://127.0.0.1:8000/health
- Status câmera: http://127.0.0.1:8000/camera/status
- Status agente: http://127.0.0.1:8000/agent/status
- Eventos: http://127.0.0.1:8000/events
- Status Ollama: http://127.0.0.1:8000/ollama/status

---

## Estrutura de arquivos

```
agrovision_ia/
├── .env                         # configurações locais
├── app.py                       # rotas FastAPI (arquivo principal)
├── requirements.txt
├── services/
│   ├── config.py                # leitura do .env e constantes
│   ├── schemas.py               # modelos Pydantic
│   ├── event_repository.py      # banco SQLite
│   ├── video_monitor.py         # câmera, YOLO, stream MJPEG
│   ├── ollama_client.py         # comunicação com Ollama
│   └── monitoring_agent.py      # agente: perfil, contexto, histórico
├── templates/
│   └── index.html               # dashboard
└── static/
    └── captures/                # imagens salvas pelo YOLO
```

---

## Perguntas para testar o agente no chat

- "O que foi detectado nos últimos eventos?"
- "Avalie o risco operacional agora."
- "Qual deve ser a próxima ação?"
- "Existe algum padrão no monitoramento?"
- "Resuma a situação da câmera em 3 pontos."

---

## Erros comuns

| Erro | Causa | Solução |
|------|-------|---------|
| Ollama não responde | Serviço não está rodando | Execute `ollama serve` |
| Modelo não encontrado | Não foi baixado | Execute `ollama pull llama3` |
| Câmera offline | Stream inacessível | Verifique CAMERA_SOURCE no .env |
| .venv não ativa | Política PowerShell | Use Set-ExecutionPolicy... |
| uvicorn não reconhecido | .venv não ativo | Ative o ambiente virtual |
