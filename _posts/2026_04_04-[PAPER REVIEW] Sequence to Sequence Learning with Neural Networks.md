---
title: "[PAPER REVIEW] Sequence to Sequence Learning with Neural Networks"
categories:
  - PAPER
---

**Deep Neural Networks (DNNs)** have shown its flexibility and power through achieving excellent performance on problems such as speech recognition and visual object recognition. But there is the limitation that it cannot be applied to problems that use sequences whose lengths are not known a-priori. To solve sequence problems, this paper presents the way of using **Long Short-Term Memory (LSTM)** architecture. One for reading the input, and the other for extracting the output based on the input sequence.  
[fig1]
  
 Before reading on, we need some background on the **Recurrent Neural Network (RNN)** and LSTM. Let's look first at the RNN.  

### 1. RNN
[fig2]
The RNN computes a sequence of output $(y_1, y_2, \dots, y_T)$s given a sequence of inputs $(x_1, x_2, \dots, x_T)$ by the following iteration.

$$
\begin{aligned}
h_t &= \sigma(W^{hx}x_t + W^{hh}h_{t-1} + b_h) \\\\
y_t &= W^{yh}h_t + b_y
\end{aligned}
$$

First, we calculate the hidden state using the input sequence. The result of the linear combination of the input and the hidden state from the previous iteration, plus some bias is pushed into the sigmoid function to complete the current hidden state. Then the current hidden state, again with some bias, is used to retrieve some output.  
[fig3]

Like we see, many forms can be induced from the original structure.  
And what about learning? We use backpropagation, specially called **Backpropagation Through Time (BPTT)**, named because of the fact that errors are propagated backwards allowing the network to learn dependencies over time.
Let's say we have the activation function $\sigma$, use softmax function for the output and use the cross-entropy loss for the output answer $y_t$ as the loss function. And we'll change the terminology a little bit.

$$
\begin{aligned}
h_t &= \sigma(W^{hx}x_t + W^{hh}h_{t-1} + b_h) \\\\
o_t &= W^{yh}h_t + b_y \\\\
\hat{y}_t &= softmax(o_t) \\\\
L &= \sum_{t=1}^T L_t
\end{aligned}
$$

The gradients $\frac{\partial L}{\partial b_y}$ and $\frac{\partial L}{\partial W^{yh}}$ can be easily derived:  

$$
\begin{aligned}
\frac{\partial L}{\partial b_y} &= \sum_{t=1}^T \frac{\partial L}{\partial \hat{y}_t} \frac{\partial \hat{y}_t}{\partial o_t} \frac{\partial o_t}{\partial b_y} = \sum_{t=1}^T (\hat{y}_t-y_t) \\\\
\frac{\partial L}{\partial W^{yh}} &= \sum_{t=1}^T \frac{\partial L}{\partial \hat{y}_t} \frac{\partial \hat{y}_t}{\partial o_t} \frac{\partial o_t}{\partial W^{yh}} = \sum_{t=1}^T (\hat{y}_t-y_t)h^{\top}_t
\end{aligned}
$$

But things get a little trickier. Say $h_t = \sigma(z_t)$, having $z_t = \sigma(W^{hx}x_t + W^{hh}h_{t-1} + b_h)$. We want the gradients $\frac{\partial L}{\partial b_h}$, $\frac{\partial L}{\partial W^{hx}}$ and $\frac{\partial L}{\partial W^{hh}}$.

$$
\begin{aligned}
\frac{\partial L}{\partial b_h} &= \sum_{t=1}^T \frac{\partial L}{\partial h_t} \frac{\partial h_t}{\partial z_t} \frac{\partial z_t}{\partial b_h} \\\\
\frac{\partial L}{\partial W^{hx}} &= \sum_{t=1}^T \frac{\partial L}{\partial h_t} \frac{\partial h_t}{\partial z_t} \frac{\partial z_t}{\partial W^{hx}} \\\\
\frac{\partial L}{\partial W^{hh}} &= \sum_{t=1}^T \frac{\partial L}{\partial h_t} \frac{\partial h_t}{\partial z_t} \frac{\partial z_t}{\partial W^{hh}} \\\\
\end{aligned}
$$

[fig4]  
As we can see, $h_t$ is used in two paths: one going to the output of current time $t$, and the other going to the next hidden layer $h_{t+1}$. When we calculate the gradient, we must keep this in mind. The transpose to the matrices have been done to match the dimensions. Note that at the first iteration, $\frac{\partial L}{\partial z_{t+1}}$ is a zero vector with a dimension of $h \mathsf{x} 1$.

$$
\begin{aligned}
\frac{\partial L}{\partial h_t} &= \frac{\partial L}{\partial o_t} \frac{\partial o_t}{\partial h_t} + \frac{\partial L}{\partial z_{t+1}} \frac{\partial z_{t+1}}{\partial y_t} = (W^{yh})^{\top} \frac{\partial L}{\partial o_t} + (W^{hh})^{\top} \frac{\partial L}{\partial z_{t+1}} \\\\
\frac{\partial L}{\partial o_t} &= \hat{y_t} - y_t, \text{ } \frac{\partial h_t}{\partial z_t} = h_t \odot (1 - h_t) \\\\
\frac{\partial L}{\partial z_t} &= \left((W^{yh})^{\top} \frac{\partial L}{\partial o_t} + (W^{hh})^{\top} \frac{\partial L}{\partial z_{t+1}}\right) \odot h_t \odot (1 - h_t) \\\\
\frac{\partial z_t}{\partial b_h} &= 1, \text{ } \frac{\partial z_t}{\partial W^{hx}} = x^{\top}_t, \text{ } \frac{\partial z_t}{\partial W^{hh}} = h^{\top}_{t-1} \\\\
\frac{\partial L}{\partial b_h} &= \sum_{t=1}^T \frac{\partial L}{\partial z_t}, \text{ } \frac{\partial L}{\partial W^{hx}} = \sum_{t=1}^T \frac{\partial L}{\partial z_t} x^{\top}_t, \text{ } \frac{\partial L}{\partial W^{hh}} = \sum_{t=1}^T \frac{\partial L}{\partial z_t} h^{\top}_{t-1}
\end{aligned}
$$

~
