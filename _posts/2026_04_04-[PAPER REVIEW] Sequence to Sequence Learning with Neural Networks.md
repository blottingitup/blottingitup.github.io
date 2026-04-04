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

Like we see, many forms can be induced from the original structure. And different forms are used to solve different problems.  
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
\frac{\partial L}{\partial h_t} &= \frac{\partial L}{\partial o_t} \frac{\partial o_t}{\partial h_t} + \frac{\partial L}{\partial z_{t+1}} \frac{\partial z_{t+1}}{\partial h_t} = (W^{yh})^{\top} \frac{\partial L}{\partial o_t} + (W^{hh})^{\top} \frac{\partial L}{\partial z_{t+1}} \\\\
\frac{\partial L}{\partial o_t} &= \hat{y_t} - y_t, \text{ } \frac{\partial h_t}{\partial z_t} = h_t \odot (1 - h_t) \\\\
\frac{\partial L}{\partial z_t} &= \left((W^{yh})^{\top} \frac{\partial L}{\partial o_t} + (W^{hh})^{\top} \frac{\partial L}{\partial z_{t+1}}\right) \odot h_t \odot (1 - h_t) \\\\
\frac{\partial z_t}{\partial b_h} &= 1, \text{ } \frac{\partial z_t}{\partial W^{hx}} = x^{\top}_t, \text{ } \frac{\partial z_t}{\partial W^{hh}} = h^{\top}_{t-1} \\\\
\frac{\partial L}{\partial b_h} &= \sum_{t=1}^T \frac{\partial L}{\partial z_t}, \text{ } \frac{\partial L}{\partial W^{hx}} = \sum_{t=1}^T \frac{\partial L}{\partial z_t} x^{\top}_t, \text{ } \frac{\partial L}{\partial W^{hh}} = \sum_{t=1}^T \frac{\partial L}{\partial z_t} h^{\top}_{t-1}
\end{aligned}
$$

Apart from the high learning cost due to the sequentiality of the network, vanishing and exploding gradient problems are key problems with RNNs. And these problems inflict damage upon the network by backpropagating improper gradient. Thus the network cannot "remember" information from distant nodes, acting as if the network has short-term memory. Although we can try to solve these problems by changing some functions ($tanh$ is more often used as the sigmoid function), it still doesn't fix the root cause.  

### 2. LSTM
[fig5]  
Now it's the LSTM, which has three "gates" which uses something called a "cell state" which is a sort of long-term memory. In contrast, the hidden state can be said to be short-term memory. The cell state and hidden state are both propagated to the next cell.

$$
\begin{aligned}
f_t &= \sigma(W_f \cdot \left[h_{t-1}, \text{ } x_t\right] + b_f)
\end{aligned}
$$

The "forget gate" uses the sigmoid function to become a kind of bias choosing which information to discard.  

$$
\begin{aligned}
i_t &= \sigma(W_i \cdot \left[h_{t-1}, \text{ } x_t\right] + b_i) \\\\
\tilde{C}_t &= \tanh(W_C \cdot \left[h_{t-1}, \text{ } x_t\right] + b_C)
\end{aligned}
$$

The "input gate" is a kind of bias that decides what information to store. The input gate has a similar form to the forget gate; they both act as some kind of bias. Meanwhile, the $\tanh$ layer gives zero-centered values to choose the information to be stored, and how. The result of this $\tanh$ layer $\tilde{C}_t$, is called "candidate value".  

$$
\begin{aligned}
C_t &= f_t \odot C_{t-1} + i_t \odot \tilde{C}_t \\\\
o_t &= \sigma(W_o \cdot \left[h_{t-1}, \text{ } x_t\right] + b_o) \\\\
h_t &= o_t \odot \tanh(C_t)
\end{aligned}
$$

Lastly, the combination of the forget gate with the cell state of previous time and the input gate with the candidate key is stored in the current cell state. It means we discard unneeded information from the previous long-term memory, and add new information to complete the current long-term memory which will be sent on to the next cell.  
We've forgotten something, the current hidden state. The "output gate" is calculated in the same manner as the forget gate and input gate. And we apply the $\tanh$ function to the cell state, and adding it to the output gate returns the value of the current hidden state. We do not want the short-term memory, the hidden state to be contaminated by the cell state. So we carefully choose the information to be the short-term memory, which acts on nearby cells.  
But why the $\tanh$ function in the output gate? The reason for $\tanh$ functions is that we don't want the data values to grow too much. We use addition to update parameters in contrast to the RNN, which used multiplication. The cell state will be kept added as time goes, and the unbounded values might cause some unwanted results. Specifically, the values being introduced in the gates will have very big absolute values. And these values go through the sigmoid function, and are saturated to regions where the gradient becomes infinitesimally small, finally resulting in gradient vanishing.

$$
\begin{aligned}
\frac{\partial L}{\partial C_t} &= \frac{\partial L}{\partial h_t} \frac{\partial h_t}{\partial C_t} + \frac{\partial L}{\partial C_{t+1}} \frac{\partial C_{t+1}}{\partial C_t} = \left(\frac{\partial L}{\partial h_t} \odot o_t \odot sech^2(C_t) \right) + \left(\frac{\partial L}{\partial C_{t+1}} \odot f_{t+1}\right)  \\\\
\frac{\partial L}{\partial h_{t-1}} &= \frac{\partial L}{\partial f_t} \frac{\partial f_t}{\partial h_{t-1}} + \frac{\partial L}{\partial i_t} \frac{\partial i_t}{\partial h_{t-1}} + \frac{\partial L}{\partial \tilde{C}_t} \frac{\partial \tilde{C}_t}{\partial h_{t-1}} + \frac{\partial L}{\partial o_t} \frac{\partial o_t}{\partial h_{t-1}}
\end{aligned}
$$

We now come back to the core part. What have we done to reduce gradient vanishing or exploding? We've already gone through the answer; the result to the gradient of the cell state doesn't have anything like the multiplication of weight matrices, which were the main culprit to the vanishing gradient problem. We can use the values preserved in long-term memory. The gradient of the hidden state uses the gradient of the cell state, which we can easily notice, instead of going back to calculate the gradient of hidden state far away.  
But unfortunately, we have only solved half of the problem. Exploding gradient still exists, if we can't get hold of the additive terms and if we have a very long sequence to process. So something called gradient norm clipping flies in to save the day.

$$
\begin{aligned}
\Vert{g}\Vert_2 > (Threshold) \Rightarrow g \leftarrow (Threshold) \cdot \frac{g}{\Vert{g}\Vert_2}
\end{aligned}
$$

We set the threshold to the L2 norm of the gradient $g$, and if the L2 norm exceedes the threshold, the size of the gradient will be reduced to one. Although we resize the gradient, the direction isn't changed. There are some ways that also changes the direction, but it is less commonly used.  

### 3. Seq2Seq
Finally, we are armed with background knowledge, ready to dive into the world of Seq2Seq.
