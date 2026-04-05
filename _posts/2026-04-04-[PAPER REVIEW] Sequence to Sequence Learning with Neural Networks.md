---
title: "[PAPER REVIEW] Sequence to Sequence Learning with Neural Networks"
categories:
  - PAPER
---

**Deep Neural Networks (DNNs)** have shown their flexibility and power through achieving excellent performance on problems such as speech recognition and visual object recognition. However, they have a limitation: they cannot be applied to problems that use sequences whose lengths are not known a-priori. To solve sequence problems, *Sequence to Sequence Learning with Neural Networks* (Sutskever et al., 2014) proposes the use of **Long Short-Term Memory (LSTM)** architecture. It consists of two components: one for reading the input, and the other for generating the output based on the input sequence.  
![Fig01: Simple Seq2Seq]({{site.baseurl}}/assets/images/260404_Fig01.jpg)  

Before reading on, we need some background on the **Recurrent Neural Network (RNN)** and LSTM. Let's look first at the RNN.  

### 1. RNN
![Fig02: RNN]({{site.baseurl}}/assets/images/260404_Fig02.png)  
The RNN computes a sequence of output $(y_1, y_2, \dots, y_T)$s given a sequence of inputs $(x_1, x_2, \dots, x_T)$ by the following iteration.

$$
\begin{aligned}
h_t &= \sigma(W^{hx}x_t + W^{hh}h_{t-1} + b_h) \\\\
y_t &= W^{yh}h_t + b_y
\end{aligned}
$$

First, we calculate the hidden state using the input sequence. The result of the linear combination of the input and the hidden state from the previous iteration, plus some bias is passed into the sigmoid function to complete the current hidden state. Then the current hidden state, again with some bias, is used to retrieve some output.  
![Fig03: Many kinds of RNNs]({{site.baseurl}}/assets/images/260404_Fig03.png)  

Like we see, many forms can be induced from the original structure. And different forms are used to solve different problems.
And what about learning? We use backpropagation, specially called **Backpropagation Through Time (BPTT)**, named because errors are propagated backwards allowing the network to learn dependencies over time.

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

But things get a little trickier. Let $h_t = \sigma(z_t)$, where $z_t = W^{hx}x_t + W^{hh}h_{t-1} + b_h$. We want the gradients $\frac{\partial L}{\partial b_h}$, $\frac{\partial L}{\partial W^{hx}}$ and $\frac{\partial L}{\partial W^{hh}}$.

$$
\begin{aligned}
\frac{\partial L}{\partial b_h} &= \sum_{t=1}^T \frac{\partial L}{\partial h_t} \frac{\partial h_t}{\partial z_t} \frac{\partial z_t}{\partial b_h} \\\\
\frac{\partial L}{\partial W^{hx}} &= \sum_{t=1}^T \frac{\partial L}{\partial h_t} \frac{\partial h_t}{\partial z_t} \frac{\partial z_t}{\partial W^{hx}} \\\\
\frac{\partial L}{\partial W^{hh}} &= \sum_{t=1}^T \frac{\partial L}{\partial h_t} \frac{\partial h_t}{\partial z_t} \frac{\partial z_t}{\partial W^{hh}} \\\\
\end{aligned}
$$

As we know, $h_t$ is used in two paths: one going to the output of current time $t$, and the other going to the next hidden layer $h_{t+1}$. When we calculate the gradient, we must keep this in mind. The matrices have been transposed to match the dimensions. Note that at the first iteration, $\frac{\partial L}{\partial z_{t+1}}$ is a zero vector with a dimension of $h \times 1$.

$$
\begin{aligned}
\frac{\partial L}{\partial h_t} &= \frac{\partial L}{\partial o_t} \frac{\partial o_t}{\partial h_t} + \frac{\partial L}{\partial z_{t+1}} \frac{\partial z_{t+1}}{\partial h_t} = (W^{yh})^{\top} \frac{\partial L}{\partial o_t} + (W^{hh})^{\top} \frac{\partial L}{\partial z_{t+1}} \\\\
\frac{\partial L}{\partial o_t} &= \hat{y_t} - y_t, \text{ } \frac{\partial h_t}{\partial z_t} = h_t \odot (1 - h_t) \\\\
\frac{\partial L}{\partial z_t} &= \left((W^{yh})^{\top} \frac{\partial L}{\partial o_t} + (W^{hh})^{\top} \frac{\partial L}{\partial z_{t+1}}\right) \odot h_t \odot (1 - h_t) \\\\
\frac{\partial z_t}{\partial b_h} &= 1, \text{ } \frac{\partial z_t}{\partial W^{hx}} = x^{\top}_t, \text{ } \frac{\partial z_t}{\partial W^{hh}} = h^{\top}_{t-1} \\\\
\frac{\partial L}{\partial b_h} &= \sum_{t=1}^T \frac{\partial L}{\partial z_t}, \text{ } \frac{\partial L}{\partial W^{hx}} = \sum_{t=1}^T \frac{\partial L}{\partial z_t} x^{\top}_t, \text{ } \frac{\partial L}{\partial W^{hh}} = \sum_{t=1}^T \frac{\partial L}{\partial z_t} h^{\top}_{t-1}
\end{aligned}
$$

Apart from the high learning cost due to the sequentiality of the network, vanishing and exploding gradient problems are key problems with RNNs. Exploding gradients destabilize the network during training by backpropagating improper gradients. Vanishing gradients makes the network unable to "remember" information from distant nodes, acting as if the network has short-term memory. Although we can try to solve these problems by changing some functions ($\tanh$ is more often used as the sigmoid function), it still doesn't fix the root cause.  

### 2. LSTM
![Fig04: LSTM]({{site.baseurl}}/assets/images/260404_Fig04.png)  
Now it's the LSTM, which has three "gates" which uses something called a "cell state" which is a sort of long-term memory. In contrast, the hidden state can be said to be short-term memory. The cell state and hidden state are both propagated to the next cell. Some extra calculations are required before obtaining $\hat{y}_t$.

$$
\begin{aligned}
f_t &= \sigma(W_f \cdot \left[h_{t-1}, \text{ } x_t\right] + b_f)
\end{aligned}
$$

The "forget gate" uses the sigmoid function to become a kind of mask choosing which information to discard. Take care not to get confused with the dimension: $W_f \cdot \left[h_{t-1}, \text{ } x_t\right] = W_{h}h_{t-1} + W_{x}x_t$.  

$$
\begin{aligned}
i_t &= \sigma(W_i \cdot \left[h_{t-1}, \text{ } x_t\right] + b_i) \\\\
\tilde{C}_t &= \tanh(W_C \cdot \left[h_{t-1}, \text{ } x_t\right] + b_C)
\end{aligned}
$$

The "input gate" is a kind of mask that decides what information to store. The input gate has a similar form to the forget gate; they both act the same way. Meanwhile, the $\tanh$ layer gives zero-centered values to choose the information to be stored, and how. The result of this $\tanh$ layer $\tilde{C}_t$, is called "candidate value".  

$$
\begin{aligned}
C_t &= f_t \odot C_{t-1} + i_t \odot \tilde{C}_t \\\\
o_t &= \sigma(W_o \cdot \left[h_{t-1}, \text{ } x_t\right] + b_o) \\\\
h_t &= o_t \odot \tanh(C_t)
\end{aligned}
$$

Lastly, the combination of the forget gate with the cell state of previous time and the input gate with the candidate value is stored in the current cell state. It means we discard unneeded information from the previous long-term memory, and add new information to complete the current long-term memory which will be sent on to the next cell.

We've forgotten something, the current hidden state. The "output gate" is calculated in the same manner as the forget gate and input gate. And we apply the $\tanh$ function to the cell state, and multiplying it element-wise to the output gate returns the value of the current hidden state. We do not want the short-term memory, the hidden state to be contaminated by the cell state. So we carefully choose the information to be the short-term memory, which acts on nearby cells.

But why the $\tanh$ function in the output gate? The reason for $\tanh$ functions is that we don't want the data values to grow too much. We use addition to update the cell state in contrast to the RNN, which used multiplication. The cell state will be kept added as time goes, and the unbounded values might cause some unwanted results. Specifically, the values being introduced in the gates will have very big absolute values. And these values go through the sigmoid function, and are saturated to regions where the gradient becomes infinitesimally small, finally resulting in gradient vanishing.

$$
\begin{aligned}
\frac{\partial L}{\partial C_t} &= \frac{\partial L}{\partial h_t} \frac{\partial h_t}{\partial C_t} + \frac{\partial L}{\partial C_{t+1}} \frac{\partial C_{t+1}}{\partial C_t} = \left(\frac{\partial L}{\partial h_t} \odot o_t \odot sech^2(C_t) \right) + \left(\frac{\partial L}{\partial C_{t+1}} \odot f_{t+1}\right)  \\\\
\frac{\partial L}{\partial h_{t-1}} &= \frac{\partial L}{\partial \hat{y}_t} \frac{\partial \hat{y}_t}{\partial h_{t-1}} + \frac{\partial L}{\partial f_t} \frac{\partial f_t}{\partial h_{t-1}} + \frac{\partial L}{\partial i_t} \frac{\partial i_t}{\partial h_{t-1}} + \frac{\partial L}{\partial \tilde{C}_t} \frac{\partial \tilde{C}_t}{\partial h_{t-1}} + \frac{\partial L}{\partial o_t} \frac{\partial o_t}{\partial h_{t-1}}
\end{aligned}
$$

We now come back to the core part. What have we done to reduce gradient vanishing or exploding? We've already gone through the answer; the result to the gradient of the cell state doesn't have anything like the multiplication of weight matrices, which were the main culprit of the vanishing gradient problem. We can use the values preserved in long-term memory. The gradient of the hidden state uses the gradient of the cell state, which we can easily notice, instead of going back to calculate the gradient of hidden state far away.

But unfortunately, we have only solved half of the problem. Exploding gradient still exists if we cannot get hold of the accumulating additive terms and if we have a very long sequence to process. To mitigate this, gradient norm clipping is practically applied.

$$
\begin{aligned}
\Vert{g}\Vert_2 > (Threshold) \Rightarrow g \leftarrow (Threshold) \cdot \frac{g}{\Vert{g}\Vert_2}
\end{aligned}
$$

We set the threshold to the L2 norm of the gradient $g$, and if the L2 norm exceedes the threshold, the size of the gradient will be reduced to one. Although we resize the gradient, the direction isn't changed. There are some ways that also changes the direction, but it is less commonly used.  

### 3. Seq2Seq
We are now armed with background knowledge, ready to dive into the world of Seq2Seq. We first look into the deep RNN structure which the authors used, proposed in *Generating Sequences With Recurrent Neural Networks* (Graves, 2014).

![Fig05: Deep LSTM]({{site.baseurl}}/assets/images/260404_Fig05.jpg)  
This structure is basically the vertically stacked version of LSTM.

$$
\begin{aligned}
h^{t}_1 &= \mathcal{H}\left(W^{h^{1}i}x_t  + W^{h^{1}h^{1}}h^{1}_{t-1} + b^{1}_h\right) \\\\
h^{t}_n &= \mathcal{H}\left(W^{h^{n}i}x_t + W^{h^{n}h^{n-1}}h^{n-1}_t + W^{h^{n}h^{n}}h^{n}_{t-1} + b^{n}_h\right) \\\\
\hat{y}_t &= b_y + \sum_{n=1}^N W^{yh^{n}}h^{n}_t \\\\
\end{aligned}
$$

The subscript orders may be a little different from the original, and $\mathcal{H}$ is the hidden layer function. The $n$-th layer of $h_t$ uses the input, $h_t$ of the $n-1$-th layer, $h_{t-1}$ of the $n$-th layer and the bias. The hidden states can store more detailed information as the layer deepens, like in Convolutional Neural Networks (CNNs). Like we see, the structure itself hasn't changed much. Also, note the "skip connections" where all the hidden states are wired so that the gradient can be calculated as the weights.

Although this idea of layering the hidden states was adopted, the skip connection mechanism was not. This means the output equation remains similar to that of the single-layered LSTM version.
Finally, we are going to look at the details of the **"Sequence-to-Sequence"** architecture.

$$
\begin{aligned}
p(y_1, \cdots, y_{T'} \vert x_1,\cdots,x_{T}) &= p(y_1 \vert v) \times p(y_2 \vert v, y_1) \times \cdots \times p(y_T' \vert v, y_1,\cdots,y_{T'-1}) \\\\
&= \prod_{t=1}^{T'} p(y_t \vert v, y_1,\cdots,y_{t-1})
\end{aligned}
$$

The conditional probability above is the main goal of which the LSTM has to estimate. The input sequence is represented by $(x_1,\cdots,x_{T})$ and $(y_1,\cdots,y_{T'})$ as the output sequence. Two LSTMs are used for this structure; **"encoder"** for the intput, the **"decoder"** for the output. The vector $v$ is a fixed-dimensional representation of the input sequence, and is given by the last hidden state of the encoder. And the decoder computes which word should be in the specific place of the sentence conditioned by the input sequence $v$, and the output sequence computed so far. Comparing with the deep RNN structure above, we only use $v$ at the first cell of the decoder as the information inside is sent on with the hidden states and cell states.

So how do we actually train this model? We've done a lot of math for training, but what happens in practice? First, we need to transform the source sentence into numbers that the network can understand. It is done using a vocabulary and an embedding matrix. For example, the authors experimented translation of English to French (WMT’14 dataset) using a fixed vocabulary each composed of 160,000 and 80,000 most used words for the source and target languages. The embedding matrix linearly projects each input word into a dense vector, where it becomes $x_t$ in the equation.

Then, we do the process of updating the cell states and hidden state all the way to the end. And when forward propagation is finished in the encoder, the vector $v$ pops out as the last hidden state. This $v$ is then sent to the start of the decoder to become the first initializing hidden state. Although not explicitly mentioned in the paper, in practice we can use another special token \<SOS\> meaning start of sentence, which is to be the first input of the decoder. The training set for the decoder; the perfectly translated sentence, becomes the input to the decoder word by word. As we've mentioned earlier, the decoder needs information about the words before the current time slot. The perfectly translated sentence becomes the ground truth for training, and is used as $y_{t-1}$ for the input in the decoder. And for this reason it is called teacher forcing. The information of words before time ${t-1}$ is saved in the hidden and cell states, so no need to give the input all the past words every time.

At each $t$, we do the same process of updating cell states and hidden states again, but this time we need an output $\hat{y}_t$ which is calculated using the hidden state in the uppermost layer as follows: $\hat{y}_t = \text{softmax}(W^{yh^{n}}h^{n}_t + b)$ . The hidden state(s) now stores the information of which word might come next, so we first linearly project it to the vocabulary and use the softmax function to retrieve the probability. And also we calculate the loss using the cross-entropy (MLE to be exact).

All of this ends at \<EOS\>, a special token indicating the end of sentence. This symbol allows the model to define a distribution over sequences of all possible lengths, overcoming the fixed-size problem. Lastly, we now calculate the total loss by summing up all the losses, and backpropagate to update the parameters.

For inference, we calculate $\hat{T} = \text{arg}\underset{T}{\text{max}} \text{ } p(T \vert S)$, where $S$ is the given source sentence and $T$ is the target sentence and $\hat{T}$ is the most likely target sentence chosen from the set of $T$. Teacher forcing is not used in inference, the actual words derived by the decoder are used instead. To find the most likely target sentence, the authors used beam search on the probabilities derived from the hidden state. A partial hypothesis serves as the prefix of some translation, and at each timestep the partial hypothesis is extended to the whole vocabulary. Beam search only maintains a "beam" a size of $B$ containing the most likely partial hypotheses and discards the rest. If the \<EOS\> comes in to the partial hypothesis, it is removed from the beam to be added to the set of complete hypotheses. The authors said a beam of size 2 provided most of the benefits of beam search.

An important thing we have to look over is that the input sentence was put into the LSTM in reverse order. The paper indicates the LSTM’s test perplexity dropped and test BLEU scores increased. By reversing the input sentence, the average distance between corresponding words in the source and target language is unchanged, but more short-term dependecies were introduced by shortening the "time lag". That is, less time and steps were consumed in calculating the first few words which induced higher accuracy and less loss in going down the network. Thus, the long-term dependency problem in long sentences was decreased significantly.

Training was done using LSTMs with four layers, SGD with no momentum and used gradient norm clipping. The authors pointed out that it is the first time that a pure neural translation system outperformed a phrase-based SMT baseline on a large scale Machine Translation (MT) task although it did not outperform the State-of-the-Art SMT systems.. The LSTM did well on long sentences but as the vector $v$, being called the "context vector" nowadays, is a fixed size vector. It means that in very long sequences, the capability of saving all the information about the input sequence might falter. This introduced the mechanism of "attention" in future studies.
