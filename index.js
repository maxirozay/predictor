const VALUE_TO_PREDICT = 1000
let data

async function run() {
  const res = await fetch(`data.json`)
  data = (await res.json())[0].map(kline => {
    return [
      parseFloat(kline[1]),
      parseFloat(kline[2]),
      parseFloat(kline[3]),
      parseFloat(kline[4])
    ]
  })

  /*const tensorData = convertToTensor(data)
  const {inputs, outputs} = tensorData

  const model = createModel(data)
  await trainModel(model, inputs, outputs) */
  const model = await tf.loadLayersModel('model/model.json')
  testModel(model, data)
  // await model.save('downloads://my-model')
}

document.addEventListener('DOMContentLoaded', run)

function testModel(model, inputData) {
  const inputTensor = tf.tensor3d([inputData.slice(0, inputData.length - VALUE_TO_PREDICT)], [1, inputData.length - VALUE_TO_PREDICT, inputData[0].length])
  const inputMax = inputTensor.max(1, true)
  const inputMin = inputTensor.min(1, true)
  const inputDiff = inputMax.sub(inputMin)
  const inputMean = inputTensor.mean(1, true)
  const normalizedInputs = inputTensor
    .sub(inputMean)
    .div(inputDiff)

  const [unNormPreds] = tf.tidy(() => {
    const preds = model.predict(normalizedInputs)

    // Un-normalize the data
    const unNormPreds = preds
      .mul(inputDiff)
      .add(inputMean)
    
    return [unNormPreds.arraySync()]
  })

  const predictedPoints = unNormPreds[0].map((val, i) => {
    return {x: inputData.length - VALUE_TO_PREDICT + i, y: val[0]}
  })
  
  const originalPoints = inputData.map((d, i) => ({
    x: i, y: d[0],
  }))
  
  tfvis.render.linechart(
    {name: 'Model Predictions vs Original Data'}, 
    {values: [originalPoints, predictedPoints], series: ['original', 'predicted']}, 
    {
      xLabel: 'time',
      yLabel: 'price',
      height: 500,
      zoomToFit: true
    }
  )
}

/* training (unused) */

function createModel() {
  // Create a sequential model
  const model = tf.sequential()
  model.add(tf.layers.lstm({units: data[0].length, inputShape: [data[0].length - VALUE_TO_PREDICT, data[0][0].length], return_sequences: true}))
  model.add(tf.layers.dense({units: VALUE_TO_PREDICT}))

  return model
}

/**
 * Convert the input data to tensors that we can use for machine 
 * learning. We will also do the important best practices of _shuffling_
 * the data and _normalizing_ the data
 */
function convertToTensor() {
  // Wrapping these calculations in a tidy will dispose any 
  // intermediate tensors.
  
  return tf.tidy(() => {
    tf.util.shuffle(data)

    const inputTensor = tf.tensor3d(data.map(d => d.slice(0, d.length - VALUE_TO_PREDICT)), [data.length, data[0].length - VALUE_TO_PREDICT, data[0][0].length])

    //ormalize the data to the range 0 - 1 using min-max scaling
    const inputMax = inputTensor.max(1, true)
    const inputMin = inputTensor.min(1, true)

    const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin))

    const outputTensor = tf.tensor2d(data.map(d => d.map(val => val[0]).slice(d.length - VALUE_TO_PREDICT)), [data.length, VALUE_TO_PREDICT])

    const outputMax = outputTensor.max()
    const outputMin = outputTensor.min()

    const normalizedOutputs = outputTensor.sub(outputMin).div(outputMax.sub(outputMin))

    return {
      inputs: normalizedInputs,
      inputMax,
      inputMin,
      outputs: normalizedOutputs,
      outputMax,
      outputMin
    }
  })  
}

async function trainModel(model, inputs, outputs) {
  // Prepare the model for training.  
  model.compile({
    optimizer: tf.train.adam(),
    loss: tf.losses.meanSquaredError,
  })
  
  const batchSize = 500
  const epochs = 1
  
  return await model.fit(inputs, outputs, {
    batchSize,
    epochs,
    shuffle: true,
    callbacks: tfvis.show.fitCallbacks(
      { name: 'Training Performance' },
      ['loss', 'mse'], 
      { height: 200, callbacks: ['onEpochEnd'] }
    )
  })
}