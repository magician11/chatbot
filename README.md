# chatbot
Learning how to code up a chat bot.

Currently studying https://chatbotsmagazine.com/contextual-chat-bots-with-tensorflow-4391749d0077

This code basically builds a model from the *intents.json* file.
Then you can feed in new sentences and it'll work out the intent of the sentence with the highest probability.
And then respond appropriately.

e.g.

```
Hi man. How are you doing?
[(u'greeting', 0.99819058)]
Hi there, how can I help?
```
