from seq2seq import tensorFromSentence, EncoderRNN, AttnDecoderRNN
from seq2seq import prepareData
import torch
from pathlib import Path


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LENGTH = 10

SOS_token = 0
EOS_token = 1


def translate(sentence, max_length=MAX_LENGTH):

    PATH = str((Path(__file__).parent).resolve())

    input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
    hidden_size = 256

    encoder = EncoderRNN(input_lang.n_words, hidden_size)
    decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1)

    encoder.load_state_dict(torch.load("%s/weights/enc.pt" % PATH))
    decoder.load_state_dict(torch.load("%s/weights/dec.pt" % PATH))

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    with torch.no_grad():

        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]
        
        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()
    
    return " ".join(decoded_words)

