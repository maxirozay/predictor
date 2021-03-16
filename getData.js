import fetch from 'node-fetch'
import fs from 'fs'

async function getData () {
  const data = []
  for (const interval of intervals) {
    for (const symbol of symbols) {
      const endTime = Date.now()
      let res = await fetch(`https://api.binance.com/api/v3/klines?symbol=${symbol}&interval=${interval.value}&endTime=${endTime - interval.ms * 1000}&limit=1000`)
      const inputs = await res.json()
      res = await fetch(`https://api.binance.com/api/v3/klines?symbol=${symbol}&interval=${interval.value}&endTime=${endTime}&limit=1000`)
      const labels = await res.json()
      const total = [...inputs, ...labels]
      if (total.length === 2000) data.push(total)
    }
  }

  fs.writeFile("data.json", JSON.stringify(data), (err) => {
    if(err) {
      return console.log(err)
    }
    console.log("The file was saved!")
  })
}

const symbols = [
  'BTCUSDT'
]
const intervals = [
  { value: '1h', ms: 3600000 }
]

getData()
