import { useState } from 'react'

interface PredictionResult {
  spam: boolean
  prob: number
}

function App() {
  const [message, setMessage] = useState('')
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    if (!message.trim()) {
      setError('Por favor, digite uma mensagem para classificar.')
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: message }),
      })

      if (!response.ok) {
        throw new Error('Erro na classificação. Tente novamente.')
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Erro desconhecido')
    } finally {
      setLoading(false)
    }
  }

  const getResultColor = (isSpam: boolean) => {
    return isSpam ? 'text-red-600' : 'text-green-600'
  }

  const getResultIcon = (isSpam: boolean) => {
    return isSpam ? '🚨' : '✅'
  }

  const getResultText = (isSpam: boolean) => {
    return isSpam ? 'Spam detectado!' : 'Não é spam'
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 py-8 px-4">
      <div className="max-w-2xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8 cursor-default">
          <h1 className="text-4xl font-bold text-gray-800 mb-2">
            🛡️ SMS Spam Classifier
          </h1>
          <p className="text-gray-600">
            Classifique mensagens de texto como spam ou não spam usando Machine Learning
          </p>
        </div>

        {/* Form */}
        <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label htmlFor="message" className="block text-sm font-medium text-gray-700 mb-2">
                Digite ou cole a mensagem:
              </label>
              <textarea
                id="message"
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                placeholder="Ex: URGENT! You have won a prize! Click here to claim..."
                className="w-full h-32 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
                disabled={loading}
              />
            </div>

            <button
              type="submit"
              disabled={loading || !message.trim()}
              className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white font-medium py-3 px-4 rounded-md transition-colors duration-200 flex items-center justify-center cursor-pointer disabled:cursor-not-allowed"
            >
              {loading ? (
                <>
                  <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Classificando...
                </>
              ) : (
                '🔍 Classificar Mensagem'
              )}
            </button>
          </form>
        </div>

        {/* Error Message */}
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-md p-4 mb-6">
            <div className="flex">
              <div className="flex-shrink-0">
                <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                </svg>
              </div>
              <div className="ml-3">
                <p className="text-sm text-red-800">{error}</p>
              </div>
            </div>
          </div>
        )}

        {/* Result */}
        {result && (
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">Resultado da Classificação</h2>

            <div className="text-center">
              <div className={`text-6xl mb-4`}>
                {getResultIcon(result.spam)}
              </div>

              <h3 className={`text-2xl font-bold mb-2 ${getResultColor(result.spam)}`}>
                {getResultText(result.spam)}
              </h3>

              <div className="bg-gray-50 rounded-lg p-4">
                <p className="text-gray-600 mb-2">Probabilidade de ser spam:</p>
                <div className="w-full bg-gray-200 rounded-full h-4">
                  <div
                    className={`h-4 rounded-full transition-all duration-500 ${result.spam ? 'bg-red-500' : 'bg-green-500'
                      }`}
                    style={{ width: `${(result.prob * 100).toFixed(1)}%` }}
                  ></div>
                </div>
                <p className="text-sm text-gray-500 mt-2">
                  {result.prob > 0.5 ?
                    `${(result.prob * 100).toFixed(1)}% de confiança` :
                    `${((1 - result.prob) * 100).toFixed(1)}% de confiança`
                  }
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Footer */}
        <div className="text-center mt-8 text-sm text-gray-500">
          <p>
            💡 <strong>Dica:</strong> Teste com mensagens como "URGENT! You won a prize!" ou "Hi, how are you?"
          </p>
          <div className="bg-yellow-100 border-l-4 border-yellow-500 text-yellow-800 p-4 rounded-md my-4 flex items-center">
            <span className="text-2xl mr-2">⚠️</span>
            <div>
              <span className="font-semibold">Atenção:</span> O classificador foi treinado apenas com mensagens em <b>inglês</b>.
              <br />
              Para obter resultados mais precisos, envie mensagens nesse idioma.
            </div>
          </div>
          <p className="mt-2">
            🔒 Por questões de privacidade, as mensagens não são armazenadas
          </p>
        </div>
      </div>
    </div>
  )
}

export default App
