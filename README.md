<br>
<br>
<br>

---
# "FaceNet: A Unified Embedding for Face Recognition and Clustering"
---

<br>
<br>

The main theme of the FaceNet paper introduced by Google researchers in the paper titled **"FaceNet: A Unified Embedding for Face Recognition and Clustering,"** focuses on the following key ideas:

### **1. Face Embedding:**
   - The primary concept of FaceNet is to convert a face image into a compact, fixed-size vector representation, known as an **embedding**. Specifically, the model reduces the facial image to a 128-dimensional vector. This vector captures the essential features of the face in a way that similar faces (e.g., images of the same person) have embeddings that are close together in this high-dimensional space, while dissimilar faces (e.g., different people) have embeddings that are far apart.

### **2. Efficient Storage and Search:**
   - The 128-dimensional vector (embedding) is small enough to allow efficient storage and search operations, even when dealing with large datasets. This compact representation is what makes FaceNet particularly powerful and practical for real-world applications.

### **3. Face Recognition:**
Face recognition is a technology that identifies or verifies individuals by analyzing their facial features. It involves capturing a face image, extracting unique features, and comparing them to a database of known faces. There are two main tasks within face recognition:

**Face Identification:**
   - The system matches a given face to a list of stored identities to determine "who" the person is. It's a one-to-many comparison where the system identifies the face from a pool of candidates.

**Face Verification:**
   - The system verifies if two face images belong to the same person. It's a one-to-one comparison, commonly used in authentication systems (e.g., unlocking a phone with facial recognition).

### **4. Face Clustering:**
Face clustering is the process of grouping a set of facial images so that images of the same person are placed in the same group (or cluster), without needing prior labels. This is useful when working with large, unlabeled datasets, as it can automatically organize images into clusters representing different individuals.

- **Unsupervised Learning**: Clustering is typically performed using unsupervised learning techniques, where the model finds patterns in the data without any labeled examples.

<br>
<br>

---
---
---

<br>
<br>

# Abstract of FaceNet:

Despite significant recent advances in the field of face
recognition [10, 14, 15, 17], implementing face verification
and recognition efficiently at scale presents serious challenges to current approaches `(এখানে, ২০১৫ সালের দিকে পেপারটি বের হয়েছে । তখন,face
recognition field এ যা কাজ হয়েছিলো সেইটা দিয়ে কখনো **at scale** Production Level এ একটা বানানো সম্ভব ছিল না ।)`. In this paper we present a
system, called FaceNet, that directly learns a mapping from
face images to a compact Euclidean space where distances
directly correspond to a measure of face similarity. Once
this space has been produced, tasks such as face recognition, verification and clustering can be easily implemented
using standard techniques with FaceNet embeddings as feature vectors.
Our method uses a `deep convolutional network(We explain it later)` trained
to directly optimize the embedding itself, rather than an intermediate bottleneck layer as in previous deep learning
approaches. To train, we use triplets of roughly aligned
matching / non-matching face patches generated using a
novel online triplet mining method. The benefit of our
approach is much greater representational efficiency: we
achieve state-of-the-art face recognition performance using
only 128-bytes per face.
On the widely used Labeled Faces in the Wild (LFW)
dataset, our system achieves a new record accuracy of
99.63%. On YouTube Faces DB it achieves 95.12%. Our
system cuts the error rate in comparison to the best published result [15] by 30% on both datasets.
We also introduce the concept of harmonic embeddings,
and a harmonic triplet loss, which describe different versions of face embeddings (produced by different networks)
that are compatible to each other and allow for direct comparison between each other.


# **Deep Convolutional Neural Network (CNN)**

### **History and Key Contributors:**

- **Yann LeCun**:
  - **LeNet-5** (1998): Developed by Yann LeCun and his colleagues, LeNet-5 was one of the earliest successful implementations of CNNs, used for digit recognition on the MNIST dataset. This work laid the groundwork for many subsequent developments in CNNs.

- **Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton**:
  - **AlexNet** (2012): This network achieved breakthrough performance in the ImageNet Large Scale Visual Recognition Challenge (ILSVRC). It demonstrated the power of deep CNNs for image classification and sparked renewed interest in deep learning.

### Summary:
While Yann LeCun's work on LeNet-5 was foundational, AlexNet is often credited with popularizing deep CNNs in the modern era due to its exceptional performance and impact on the field. Both contributions were crucial in the development and success of deep convolutional neural networks.


----

<br>
<br>
<br>

# Motivation Problems With Old Appraoaches:


![phto_one](note_image/pic01.png)



### 1. **Verification**

**Definition:**
- Verification refers to confirming whether a given photo matches a specific person. In other words, it answers the question, "Is this person who they claim to be?",`In the photo, this is Tess or not.` 

**Application:**
- **Example:** If you have a photo of someone and want to verify if this person is indeed the one shown in the photo, verification is used. This process involves comparing the photo against a known identity to check for a match.


### 2. **Identification**

**Definition:**
- Identification involves determining who a person is from a photo when their identity is unknown. It answers the question, "Who is this person?"`In the picture, is there is present Tess and Anders.`

**Application:**
- **Example:** Given a random photo of a person, the system identifies which person in the database this photo most likely represents.


### 3. **Clustering**

**Definition:**
- Clustering is the process of grouping together images of the same person, even if the images have different clothing or appearances. This process helps in organizing images based on similarities.

**Application:**
- **Example:** If you have multiple photos of the same person but with different clothes, clustering helps group these images together so that you can manage or analyze them as belonging to the same individual.

---

<br>

#### **Current Challenges in Face Verification and Identification (Before facenet came):**


![image_two](note_image/pic02.png)

#### `ধরি আমাদের একটা startup আছে । সেইখানে, উপরের ছবিতে দেখানো সবাই আছে। যদি আমরা Classification ব্যবহার করে একটা face recognition system তৈরি করি এই startup জন্য, যখন  Tess যাবে তখন সে যেতে পারবে । কিন্তু কোন এলে যেতে পারবে না । যদি আমাদের company microsoft or google এর মতো বড় হয় সেক্ষেত্রেঃ  `

1. **Scalability Issues with Traditional Classification:**
   - **Problem:** Traditional image classification methods require a separate class for each individual. For instance, in a large company with 100,000 employees, this means creating and managing 100,000 classes.
   - **Limitation:** This approach is impractical due to the sheer number of classes and images required. It also struggles with accurately classifying individuals who do not have many images available.

#### `যদি কোন একজন ব্যক্তি আমাদের stratup এ join করে তাহলেঃ `

2. **Adaptability Problems:**
   - **Problem:** If a new person joins the company, the network needs to be retrained to include this new individual. This is not feasible for dynamic environments with frequent changes.

#### `আমাদের,Tim শুধু শুধুই অনেক selfie নেই। অর্থাৎ, এক্ষেত্রে,Tim কে ভালোভাবে classified করা যাবে, কারণ তার অনেক ছবি রয়েছে ।  `

![image03](note_image/pic03.png)


3. **Poor Performance with Sparse Data:**
   - **Problem:** Individuals with fewer images, like those who do not take many selfies, may not be classified accurately. This results in poor performance for less-represented individuals.


#### `এখন আমরা কি করতে পারি যেহেতু আমরা, classification use করতে পারবো না । ML থেকে, আমরা জানি Clustering( that group similar thing into a same group)। সেইটা কি করা যাবে? হ্যাঁ, আমরা এমন কিছুই শিখবো । যেইটাকে আমরা similarity function বলতেছি । একটা জিনিসের সাথে আরেকটা জিনিস কতটা similar সেইটা বের করতে পারবো ।`


Now, We're gonna learn a similarity function. We're gonna instead of like actually doing classification, we're gonna try to figure out how similar things are like or how similar these faces are and if they're similar enough or it can identify them as true. So, in order to do that we need to figure out these 128 byte embeddings. If you don't know what embeddings are then embeddings are sort of like, the core of what deep learning is all about finding a semantic meaning like finding an actual meaningful way of describing things so if we look at words for example: , `এখানে ৩টা word দিয়ে প্রথমে word embeddings বুঝানো হয়েছে । যেইটার example আমি সবসময় দিয়ে থাকি, interesting। `


# Here’s a comprehensive list of both word and image embedding techniques:

### Word Embedding Techniques

1. **One-Hot Encoding**
2. **Ordinal Encoding**
3. **Bag of Words (BoW)**
4. **TF-IDF (Term Frequency-Inverse Document Frequency)**
5. **Bag of N-Grams**
6. **Word2Vec** (Skip-gram, CBOW)
7. **GloVe (Global Vectors for Word Representation)**
8. **FastText**
9. **ELMo (Embeddings from Language Models)**
10. **BERT (Bidirectional Encoder Representations from Transformers)**
11. **GPT (Generative Pre-trained Transformer)**
12. **Transformer Embeddings**
13. **T5 (Text-To-Text Transfer Transformer)**
14. **RoBERTa (Robustly Optimized BERT Approach)**
15. **XLNet**
16. **DistilBERT**

### Image Embedding Techniques

1. **Raw Pixels**
2. **Histogram of Oriented Gradients (HOG)**
3. **Scale-Invariant Feature Transform (SIFT)**
4. **Speeded-Up Robust Features (SURF)**
5. **Convolutional Neural Networks (CNNs)**:
   - **AlexNet**
   - **VGGNet**
   - **ResNet**
   - **Inception**
   - **DenseNet**
6. **Pre-trained Networks**:
   - **ImageNet Models**
7. **Feature Extraction**:
   - **GloVe for Images** (using text-based features)
8. **Autoencoders**
9. **Generative Adversarial Networks (GANs)**
10. **Variational Autoencoders (VAEs)**
11. **Deep Embeddings** (using neural networks)
12. **CLIP (Contrastive Language-Image Pre-training)**
13. **Swin Transformer**
14. **ViT (Vision Transformer)**
15. **DeiT (Data-efficient Image Transformer)**


<br>
<br>
<br>

---

# Let's fine simillary of movies and For image embedding can we use simple CNN:

<br>
<br>
<br>

