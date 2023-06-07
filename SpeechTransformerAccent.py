import os
from glob import glob
import tensorflow as tf
from tensorflow import keras
from keras import layers
import ssl
import re


class TokenEmbedding(layers.Layer):
    def __init__(self, num_vocab=1000, maxlen=100, num_hid=64):
        super().__init__()
        self.emb = tf.keras.layers.Embedding(num_vocab, num_hid)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=num_hid)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        x = self.emb(x)
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        return x + positions


class SpeechFeatureEmbedding(layers.Layer):
    def __init__(self, num_hid=64, maxlen=100):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv1D(
            num_hid, 11, strides=2, padding="same", activation="relu"
        )
        self.conv2 = tf.keras.layers.Conv1D(
            num_hid, 11, strides=2, padding="same", activation="relu"
        )
        self.conv3 = tf.keras.layers.Conv1D(
            num_hid, 11, strides=2, padding="same", activation="relu"
        )
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=num_hid)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.conv3(x)


class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, rate=0.2):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [
                layers.Dense(feed_forward_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, dropout_rate=0.1):
        super().__init__()
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)
        self.self_att = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.enc_att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.self_dropout = layers.Dropout(0.5)
        self.enc_dropout = layers.Dropout(0.2)
        self.ffn_dropout = layers.Dropout(0.2)
        self.ffn = keras.Sequential(
            [
                layers.Dense(feed_forward_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )

    def causal_attention_mask(self, batch_size, n_dest, n_src, dtype):
        """Masks the upper half of the dot product matrix in self attention.

        This prevents flow of information from future tokens to current token.
        1's in the lower triangle, counting from the lower right corner.
        """
        i = tf.range(n_dest)[:, None]
        j = tf.range(n_src)
        m = i >= j - n_src + n_dest
        mask = tf.cast(m, dtype)
        mask = tf.reshape(mask, [1, n_dest, n_src])
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], 0
        )
        return tf.tile(mask, mult)

    def call(self, enc_out, target):
        input_shape = tf.shape(target)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        causal_mask = self.causal_attention_mask(batch_size, seq_len, seq_len, tf.bool)
        target_att = self.self_att(target, target, attention_mask=causal_mask)
        target_norm = self.layernorm1(target + self.self_dropout(target_att))
        enc_out = self.enc_att(target_norm, enc_out)
        enc_out_norm = self.layernorm2(self.enc_dropout(enc_out) + target_norm)
        ffn_out = self.ffn(enc_out_norm)
        ffn_out_norm = self.layernorm3(enc_out_norm + self.ffn_dropout(ffn_out))
        return ffn_out_norm


class Transformer(keras.Model):
    def __init__(
            self,
            num_hid=64,
            num_head=2,
            num_feed_forward=128,
            source_maxlen=100,
            target_maxlen=100,
            num_layers_enc=4,
            num_layers_dec=1,
            num_classes=10,
            dropout_rate=0.2,
    ):
        super().__init__()
        self.loss_metric = keras.metrics.Mean(name="loss")
        self.num_layers_enc = num_layers_enc
        self.num_layers_dec = num_layers_dec
        self.target_maxlen = target_maxlen
        self.num_classes = num_classes

        self.enc_input = SpeechFeatureEmbedding(num_hid=num_hid, maxlen=source_maxlen)
        self.dec_input = TokenEmbedding(
            num_vocab=num_classes, maxlen=target_maxlen, num_hid=num_hid
        )

        self.encoder = keras.Sequential(
            [self.enc_input]
            + [
                TransformerEncoder(num_hid, num_head, num_feed_forward, rate=dropout_rate)
                for _ in range(num_layers_enc)
            ]
        )

        for i in range(num_layers_dec):
            setattr(
                self,
                f"dec_layer_{i}",
                TransformerDecoder(num_hid, num_head, num_feed_forward, dropout_rate=dropout_rate),
            )

        self.classifier = layers.Dense(num_classes)

    def decode(self, enc_out, target):
        y = self.dec_input(target)
        for i in range(self.num_layers_dec):
            y = getattr(self, f"dec_layer_{i}")(enc_out, y)
        return y

    def call(self, inputs):
        source = inputs[0]
        target = inputs[1]
        x = self.encoder(source)
        y = self.decode(x, target)
        return self.classifier(y)

    @property
    def metrics(self):
        return [self.loss_metric]

    def train_step(self, batch):
        """Processes one batch inside model.fit()."""
        source = batch["source"]
        target = batch["target"]
        dec_input = target[:, :-1]
        dec_target = target[:, 1:]
        with tf.GradientTape() as tape:
            preds = self([source, dec_input])
            one_hot = tf.one_hot(dec_target, depth=self.num_classes)
            mask = tf.math.logical_not(tf.math.equal(dec_target, 0))
            loss = self.compiled_loss(one_hot, preds, sample_weight=mask)
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.loss_metric.update_state(loss)
        return {"loss": self.loss_metric.result()}

    def test_step(self, batch):
        source = batch["source"]
        target = batch["target"]
        dec_input = target[:, :-1]
        dec_target = target[:, 1:]
        preds = self([source, dec_input])
        one_hot = tf.one_hot(dec_target, depth=self.num_classes)
        mask = tf.math.logical_not(tf.math.equal(dec_target, 0))
        loss = self.compiled_loss(one_hot, preds, sample_weight=mask)
        self.loss_metric.update_state(loss)
        return {"loss": self.loss_metric.result()}

    def generate(self, source, target_start_token_idx):
        """Performs inference over one batch of inputs using greedy decoding."""
        bs = tf.shape(source)[0]
        enc = self.encoder(source)
        dec_input = tf.ones((bs, 1), dtype=tf.int32) * target_start_token_idx
        dec_logits = []
        for i in range(self.target_maxlen - 1):
            dec_out = self.decode(enc, dec_input)
            logits = self.classifier(dec_out)
            logits = tf.argmax(logits, axis=-1, output_type=tf.int32)
            last_logit = tf.expand_dims(logits[:, -1], axis=-1)
            dec_logits.append(last_logit)
            dec_input = tf.concat([dec_input, last_logit], axis=-1)
        return dec_input


def get_phoneme_files(wav_directory, txt_directory):
    # Iterate over files with .wav extension in the wav_directory
    for filename in os.listdir(wav_directory):
        if filename.endswith(".wav"):
            wav_filepath = os.path.join(wav_directory, filename)
            txt_filepath = os.path.join(txt_directory,
                                        filename[:-4] + ".txt")  # Replaces .wav with .txt in the filename

            if os.path.isfile(txt_filepath):
                # Read the .txt file
                with open(txt_filepath, "r") as txt_file:
                    txt_contents = txt_file.read()
                    txt_inner = re.findall(r'\[(.*?)\]', txt_contents)
                    id_to_text[filename[:-4]] = txt_inner


def get_data(wavs, id_to_text, maxlen=50):
    """ returns mapping of audio paths and transcription texts """
    data = []
    for w in wavs:
        id = w.split("/")[-1].split(".")[0]
        if (id not in id_to_text):
            continue
        if len(id_to_text[id]) < maxlen:
            data.append({"audio": w, "text": id_to_text[id]})

    return data


class VectorizeChar:
    def __init__(self, max_len=50):
        self.vocab = (
                ["-", "#", "<", ">"]
                + ['̚', 'M', 'r', 'ĩ', 'ɚ', 'ʈ', 'ʌ', '̥', 'I', '̺', 'ɑ', '6', 'ɖ', 'e', 'ɞ', '͂', '̹', 'Y', 'ʔ',
                   '\xa0', 'œ', 'ɘ', 'ᶴ', 'y', 'S', 'ˠ', 'ɹ', '̘', 'ː', 'z', '̞', 'ɪ', 'C', 'ə', ':', 'ʀ', '[', '7',
                   'V', 'ˡ', 'g', 'ɬ', '̑', 'ʤ', 'ø', 'ʒ', '٤', 'A', 's', 'E', '-', 'õ', 'H', 'ɝ', 'ʁ', 'ɨ', 'h', 'ũ',
                   '̟', 'j', 'ɻ', 'ӕ', 'ᵊ', '̃', 'P', '̮', '͡', '̰', 'ʍ', 'ǝ', 'β', 'd', 'ʊ', '˞', '̪', 'ˤ', 'ǀ', 'ʉ',
                   'm', 'ŭ', '\uf1c8', '̝', 'J', 'ˀ', 'ɡ', 'æ', '̜', 'O', 't', 'ɮ', '̱', 'ɱ', 'ت', '3', 'θ', 'L', 'N',
                   'ˢ', 'ɦ', 'ɺ', 'p', 'ʂ', '˳', ' ', 'ـ', 'ʏ', 'ʎ', 'G', 'W', '\n', 'ɲ', 'ʟ', '̠', 'k', 'ç', '̀', 'ʧ',
                   'ʐ', 'é', 'ɓ', 'D', 'ɗ', '˺', 'ɵ', 'c', '͉', 'ɸ', 'χ', '0', 'u', ']', '8', 'ɰ', 'ã', '2', 'U', 'ɳ',
                   'R', 'ʑ', ',', '̩', 'ɒ', 'B', 'ð', '͎', 'a', 'l', '͆', 'ˈ', 'ʲ', 'ɛ', 'f', 'ɐ', 'ɾ', '∫', 'x', '1',
                   'ɤ', 'Q', 'ʷ', 'ʝ', 'ʃ', 'Z', '̤', 'ɜ', '9', 'ˑ', 'ẽ', 'ŋ', 'ǂ', 'ⅼ', '̆', 'ɭ', 'ɔ', 'ʕ', 'ɯ', '\t',
                   '̬', '5', 'n', '4', 'K', 'v', '̻', 'F', '̌', '̙', 'ñ', 'ă', 'ʰ', 'T', 'ɽ', 'ɣ', 'w', 'b', 'ʋ', 'ɕ',
                   'i', 'ә', 'o']
                + [" ", ".", ",", "?"]
        )
        self.max_len = max_len
        self.char_to_idx = {}
        for i, ch in enumerate(self.vocab):
            self.char_to_idx[ch] = i

    def __call__(self, text):
        if len(text) == 0:
            text = ""
        else:
            text = text[0]
        text = text.lower()
        text = text[: self.max_len - 2]
        text = "<" + text + ">"
        pad_len = self.max_len - len(text)
        return [self.char_to_idx.get(ch, 1) for ch in text] + [0] * pad_len

    def get_vocabulary(self):
        return self.vocab


def create_text_ds(data):
    texts = [_["text"] for _ in data]
    text_ds = [vectorizer(t) for t in texts]
    text_ds = tf.data.Dataset.from_tensor_slices(text_ds)
    return text_ds


def path_to_audio(path):
    # spectrogram using stft
    audio = tf.io.read_file(path)
    audio, _ = tf.audio.decode_wav(audio, 1)
    audio = tf.squeeze(audio, axis=-1)
    stfts = tf.signal.stft(audio, frame_length=200, frame_step=80, fft_length=256)
    x = tf.math.pow(tf.abs(stfts), 0.5)
    # normalisation
    means = tf.math.reduce_mean(x, 1, keepdims=True)
    stddevs = tf.math.reduce_std(x, 1, keepdims=True)
    x = (x - means) / stddevs
    audio_len = tf.shape(x)[0]

    pad_len = 9600
    paddings = tf.constant([[0, pad_len], [0, 0]])
    x = tf.pad(x, paddings, "CONSTANT")[:pad_len, :]
    return x


def create_audio_ds(data):
    flist = [_["audio"] for _ in data]
    audio_ds = tf.data.Dataset.from_tensor_slices(flist)
    audio_ds = audio_ds.map(
        path_to_audio, num_parallel_calls=tf.data.AUTOTUNE
    )
    return audio_ds


def create_tf_dataset(data, bs=4):
    audio_ds = create_audio_ds(data)
    text_ds = create_text_ds(data)
    ds = tf.data.Dataset.zip((audio_ds, text_ds))
    ds = ds.map(lambda x, y: {"source": x, "target": y})
    ds = ds.batch(bs)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


class DisplayOutputs(keras.callbacks.Callback):
    def __init__(
            self, batch, idx_to_token, target_start_token_idx=27, target_end_token_idx=28
    ):
        """Displays a batch of outputs after every epoch

        Args:
            batch: A test batch containing the keys "source" and "target"
            idx_to_token: A List containing the vocabulary tokens corresponding to their indices
            target_start_token_idx: A start token index in the target vocabulary
            target_end_token_idx: An end token index in the target vocabulary
        """
        self.batch = batch
        self.target_start_token_idx = target_start_token_idx
        self.target_end_token_idx = target_end_token_idx
        self.idx_to_char = idx_to_token

    def on_epoch_end(self, epoch, logs=None):

        if epoch % 5 != 0:
            return

        checkpoint_manager.save()

        source = self.batch["source"]
        target = self.batch["target"].numpy()
        bs = tf.shape(source)[0]
        preds = self.model.generate(source, self.target_start_token_idx)
        preds = preds.numpy()
        for i in range(bs):
            target_text = "".join([self.idx_to_char[_] for _ in target[i, :]])
            prediction = ""
            for idx in preds[i, :]:
                prediction += self.idx_to_char[idx]
                if idx == self.target_end_token_idx:
                    break
            print(f"target:     {target_text.replace('-', '')}")
            print(f"prediction: {prediction}\n")


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=5000):
        super().__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        return {
            'd_model': self.d_model,
            'warmup_steps': self.warmup_steps
        }


transcripts_path = "/Speech/processed transcripts1"
saveto = "/Speech/processed wav files1"

# transcripts_path = "/home/ubuntu/cs230/processed transcripts1"
# saveto = "/home/ubuntu/cs230/processed wav files1"

id_to_text = {}
get_phoneme_files(saveto, transcripts_path)

wavs = keys_list = [saveto + "/" + key + ".wav" for key in id_to_text.keys()]

max_target_len = 500
data = get_data(wavs, id_to_text, max_target_len)
vectorizer = VectorizeChar(max_target_len)
print("vocab size", len(vectorizer.get_vocabulary()))

split = int(len(data) * 0.99)
train_data = data[:split]
test_data = data[split:]
ds = create_tf_dataset(train_data, bs=64)
val_ds = create_tf_dataset(test_data, bs=4)

batch = next(iter(val_ds))

# The vocabulary to convert predicted indices into characters
idx_to_char = vectorizer.get_vocabulary()
display_cb = DisplayOutputs(batch, idx_to_char, target_start_token_idx=2, target_end_token_idx=3)

model = Transformer(
    num_hid=200,
    num_head=4,
    num_feed_forward=256,
    target_maxlen=max_target_len,
    num_layers_enc=6,
    num_layers_dec=6,
    num_classes=208,
)
loss_fn = tf.keras.losses.CategoricalCrossentropy(
    from_logits=True, label_smoothing=0.1,
)

d_model = 512
learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.legacy.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

checkpoint = tf.train.Checkpoint(transformer_model=model, optimizer=optimizer)
checkpoint_manager = tf.train.CheckpointManager(checkpoint, '/Users/oleg1/Desktop/Stanford/Checkpoints/', max_to_keep=5)


history = model.fit(ds, validation_data=val_ds, callbacks=[display_cb], epochs=100)

# Save the model in the SavedModel format
model.save_weights('ModelWeights')

