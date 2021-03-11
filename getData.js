import fetch from 'node-fetch'
import fs from 'fs'

async function getData () {
  const data = []
  const startTime = new Date('2017-09-01').getTime()
  const endTime = Date.now()
  for (let time = startTime; time < endTime; time += 18144000000) {
    for (const interval of intervals) {
      for (const symbol of symbols) {
        const res = await fetch(`https://api.binance.com/api/v3/klines?symbol=${symbol}&interval=${interval}&endTime=${time}&limit=600`)
        const json = await res.json()
        if (json.length === 600) data.push(json)
      }
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
  'BTCUSDT',
  'ETHUSDT',
  'XRPUSDT',
  'BNBUSDT',
  'ADAUSDT',
  'DOTUSDT',
  'IOTAUSDT',
  'MATICUSDT',
  'CHRUSDT',
  'CHZUSDT',
  'VITEUSDT',
  'SXPUSDT'
]
const intervals = [
  '5m',
  '30m',
  '1h',
  '6h',
  '12h',
  '1d',
  '1w'
]

getData()
