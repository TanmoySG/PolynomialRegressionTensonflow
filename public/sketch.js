let a, b, c, d, dragging = false, x_vals = [], y_vals = [];

const learningRate = 0.2;
const optimizer = tf.train.adam(learningRate);

function predict(xs) {
	return xs.pow(tf.scalar(3)).mul(a).add(xs.square().mul(b)).add(xs.mul(c)).add(d);
}

function loss(predicts, labels) {
	return predicts.sub(labels).square().mean();
}
function mousePressed() {
	dragging = true;
}

function mouseReleased() {
	dragging = false;
}

function setup() {
	createCanvas(800, 800);
	a = tf.variable(tf.scalar(random(-1, 1)));
	b = tf.variable(tf.scalar(random(-1, 1)));
	c = tf.variable(tf.scalar(random(-1, 1)));
	d = tf.variable(tf.scalar(random(-1, 1)));
}

function draw() {
	// console.log(mouseX, mouseY);
	if (dragging) {
		let pointX = map(mouseX, 0, width, -1, 1);
		let pointY = map(mouseY, 0, height, 1, -1);
		x_vals.push(pointX);
		y_vals.push(pointY);
		//console.log(targetX, targetY);
	} else {
		tf.tidy(() => {
			const ys = tf.tensor1d(y_vals);
			optimizer.minimize(() => loss(predict(tf.tensor1d(x_vals)), ys));
		});
	}
	background(0);

	stroke(255);
	strokeWeight(20);
	for (let i = 0; i < x_vals.length; i++) {
		let px = map(x_vals[i], -1, 1, 0, width);
		let py = map(y_vals[i], -1, 1, height, 0);
		point(px, py);
	}

	const curveX = [];
	for (let x = -1; x <= 1; x += 0.05) {
		curveX.push(x);
	}

	const ys = tf.tidy(() => predict(tf.tensor1d(curveX)));
	let curveY = ys.dataSync();
	ys.dispose();

	beginShape();
	noFill();
	stroke(255);
	strokeWeight(2);
	for (let i = 0; i < curveX.length; i++) {
		let x = map(curveX[i], -1, 1, 0, width);
		let y = map(curveY[i], -1, 1, height, 0);
		vertex(x, y);
	}
	endShape();
	console.log(tf.memory().numTensors);
}
