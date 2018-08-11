"""MIT License

Copyright (c) 2018 dogydev

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""

from random import randrange


class SLP(object):

    def __init__(self, lr, n):

        self.bias = 0
        self.n = n
        self.weights = [None] * n

        for i in range(len(self.weights)):
            self.weights[i] = randrange(-1, 1)

        self.lr = lr

    def activate(self, g):

        if g > 0:
            return 1
        else:
            return -1

    def feedforward(self, inputs):

        for i in range(len(self.weights)):
            self.bias += inputs[i] * self.weights[i]

        return self.activate(self.bias)

    def train(self, inputs, target):

        inputnumbers = self.feedforward(inputs)
        error = target - inputnumbers

        print(error)

        for i in range(len(self.weights)):
            self.weights[i] += self.lr * error * inputs[i]

            print("Weights:")
            print(self.weights)
