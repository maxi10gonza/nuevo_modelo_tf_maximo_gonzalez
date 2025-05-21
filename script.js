let modelo;

async function entrenarModelo() {
    const estado = document.getElementById('estadoEntrenamiento');
    estado.textContent = "⏳ Entrenando modelo...";

    // Datos: y = 2x² - 3x + 1
    const xs = tf.linspace(-10, 10, 200);
    const ys = xs.square().mul(2).sub(xs.mul(3)).add(1);

    const xEntrenamiento = ys;
    const yEntrenamiento = xs;

    modelo = tf.sequential();
    modelo.add(tf.layers.dense({ inputShape: [1], units: 16, activation: 'relu' }));
    modelo.add(tf.layers.dense({ units: 16, activation: 'relu' }));
    modelo.add(tf.layers.dense({ units: 1 }));

    modelo.compile({
        optimizer: tf.train.adam(0.01),
        loss: 'meanSquaredError'
    });

    await modelo.fit(xEntrenamiento.reshape([200, 1]), yEntrenamiento.reshape([200, 1]), {
        epochs: 300,
        callbacks: {
        onEpochEnd: (epoch, logs) => {
            if (epoch % 50 === 0) {
            estado.textContent = `Epoch ${epoch} - Pérdida: ${logs.loss.toFixed(4)}`;
            }
        }
        }
    });

    estado.textContent = "✅ Modelo entrenado correctamente.";
}

function predecirX() {
    const resultado = document.getElementById('resultado');
    const valorY = parseFloat(document.getElementById('valorY').value);

    if (!modelo) {
        resultado.textContent = "Primero debes entrenar el modelo.";
        return;
    }

    const entrada = tf.tensor2d([[valorY]]);
    const prediccion = modelo.predict(entrada);

    prediccion.array().then(arr => {
        const valorX = arr[0][0].toFixed(4);
        resultado.textContent = `Predicción: para y = ${valorY}, x = ${valorX}`;
    });
}
