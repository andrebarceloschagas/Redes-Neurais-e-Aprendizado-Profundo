# Adaptações do Notebook CNN - De Cães e Gatos para Tumores Cerebrais

Este documento descreve as adaptações realizadas para transformar o notebook `cnn_caes_gatos.ipynb` (classificação de cães e gatos) no notebook `cnn_brain_tumor.ipynb` (classificação de tumores cerebrais em imagens de MRI).

---

## 📋 Resumo das Mudanças

| Aspecto | Cães e Gatos | Tumores Cerebrais |
|---------|--------------|-------------------|
| **Classes** | 2 (dog, cat) | 4 (glioma, meningioma, notumor, pituitary) |
| **Tamanho das imagens** | 100x100 | 150x150 |
| **Carregamento de dados** | Manual (listdir + load_img) | `keras.utils.image_dataset_from_directory()` |
| **Data Augmentation** | Não utilizado | Sim (camadas Keras: RandomRotation, RandomZoom, etc.) |
| **Arquitetura CNN** | Não definida no notebook | 5 camadas convolucionais + BatchNorm + Dropout |
| **Callbacks** | Não utilizados | EarlyStopping, ModelCheckpoint, ReduceLROnPlateau |
| **Métricas** | Básicas | Matriz de confusão, relatório de classificação |
| **APIs Keras** | Legado (`tensorflow.keras`) | Moderno (`keras` 3.x + `tf.data`) |

---

## 🔄 Adaptações Detalhadas

### 1. **Estrutura do Dataset**

**Antes (Cães e Gatos):**
```
imgsdogsandcats/
├── dogs/
└── cats/
```

**Depois (Tumores Cerebrais):**
```
Brain Tumor MRI Dataset/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

**Justificativa:** O dataset de tumores cerebrais possui uma estrutura mais organizada com separação explícita entre treino e teste, além de 4 classes em vez de 2.

---

### 2. **Carregamento de Dados**

**Antes:**
- Carregamento manual usando `os.listdir()` e `keras.utils.load_img()`
- Salvamento em arquivos `.pkl` e `.csv`
- Processamento manual de strings para converter dados

**Depois:**
- Uso do `keras.utils.image_dataset_from_directory()` (API moderna)
- Pipeline `tf.data` para carregamento eficiente
- Otimização com `cache()`, `shuffle()` e `prefetch()`
- Split automático treino/validação (80/20)

```python
# Novo método de carregamento (API moderna)
train_dataset = keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="training",
    seed=42,
    label_mode='categorical',
    shuffle=True
)

# Otimização com tf.data
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
```

**Justificativa:** 
- `image_dataset_from_directory()` é a API recomendada no Keras 3.x
- `tf.data` oferece melhor performance e gerenciamento de memória
- `ImageDataGenerator` está depreciado

---

### 3. **Data Augmentation**

**Antes:** Não utilizado

**Depois:** Implementado com camadas Keras (API moderna):
```python
# Camadas de augmentation integradas ao modelo
data_augmentation = keras.Sequential([
    layers.RandomRotation(0.1),           # Rotação aleatória (±10% de 360°)
    layers.RandomTranslation(0.2, 0.2),   # Deslocamento horizontal e vertical
    layers.RandomZoom(0.2),               # Zoom aleatório
    layers.RandomFlip("horizontal"),      # Flip horizontal
], name="data_augmentation")
```

**Justificativa:** 
- As camadas de augmentation são aplicadas automaticamente apenas durante o treinamento
- Integração nativa com o modelo Keras
- Substituem o depreciado `ImageDataGenerator`
- Melhor performance com GPU

---

### 4. **Arquitetura do Modelo CNN**

**Antes:** O notebook original não define explicitamente a arquitetura CNN (foca apenas no carregamento de dados)

**Depois:** CNN completa com 5 blocos convolucionais usando API Funcional:

```python
# Modelo usando API Funcional do Keras (moderno)
inputs = keras.Input(shape=input_shape)

# Data augmentation integrado ao modelo
x = data_augmentation(inputs)

# Bloco 1: 32 filtros
x = layers.Conv2D(32, (3, 3), activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2, 2))(x)

# ... (blocos 2-5 com 64, 128, 256, 512 filtros)

# Camadas densas com Dropout
x = layers.Flatten()(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.3)(x)

# Camada de saída
outputs = layers.Dense(num_classes, activation='softmax')(x)

model = keras.Model(inputs, outputs)
```

**Justificativa:** 
- **API Funcional:** Permite integrar data augmentation diretamente no modelo
- **BatchNormalization:** Estabiliza e acelera o treinamento
- **Dropout:** Previne overfitting nas camadas densas
- **Softmax:** Apropriado para classificação multiclasse (4 classes)

---

### 5. **Função de Perda e Métricas**

**Antes:** Classificação binária (2 classes - dog/cat)

**Depois:** Classificação multiclasse (4 classes)
```python
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',  # Para multiclasse
    metrics=['accuracy']
)
```

**Justificativa:** `categorical_crossentropy` é a função de perda adequada para problemas de classificação multiclasse com labels one-hot encoded.

---

### 6. **Callbacks de Treinamento**

**Antes:** Não utilizados

**Depois:** Três callbacks importantes:

```python
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    ),
    ModelCheckpoint(
        'best_brain_tumor_model.keras',
        monitor='val_accuracy',
        save_best_only=True
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-7
    )
]
```

**Justificativa:**
- **EarlyStopping:** Para o treinamento quando não há melhora, evitando overfitting
- **ModelCheckpoint:** Salva o melhor modelo durante o treinamento
- **ReduceLROnPlateau:** Reduz a taxa de aprendizado quando o modelo estagna

---

### 7. **Avaliação e Métricas**

**Antes:** Apenas visualização básica das imagens

**Depois:** Avaliação completa com:
- Matriz de confusão com visualização heatmap
- Relatório de classificação (precision, recall, F1-score)
- Métricas por classe (acurácia, precisão, recall, F1)
- Visualização de predições com confiança

```python
# Relatório de classificação
print(classification_report(true_classes, predicted_classes, target_names=target_names))

# Matriz de confusão
cm = confusion_matrix(true_classes, predicted_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
```

**Justificativa:** Em aplicações médicas, é crucial entender não apenas a acurácia geral, mas também o desempenho por classe e os tipos de erros cometidos.

---

### 8. **Função de Predição Individual**

**Antes:** Não disponível

**Depois:** Função para classificar uma única imagem:
```python
def predict_single_image(model, image_path, img_size=(150, 150)):
    # Carrega, preprocessa e prediz uma imagem
    # Retorna a classe predita e a confiança
    # Exibe probabilidades para cada classe
```

**Justificativa:** Útil para uso prático do modelo em produção ou para testar imagens individuais.

---

### 9. **Salvamento do Modelo**

**Antes:** Dados salvos em arquivos `.pkl` e `.csv`

**Depois:** Modelo salvo apenas no formato moderno `.keras`:
```python
model.save('brain_tumor_classifier_final.keras')  # Formato nativo Keras 3.x
```

**Justificativa:** 
- O formato `.keras` é o padrão recomendado no Keras 3.x
- O formato `.h5` está depreciado e pode ter problemas de compatibilidade
- Melhor suporte para camadas customizadas e configurações complexas

---

## 🔄 Atualização para APIs Modernas do Keras

O notebook foi atualizado para usar as APIs mais recentes do Keras 3.x, removendo dependências legadas:

### Mudanças nas Importações

**Antes (Legado):**
```python
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array
```

**Depois (Moderno):**
```python
import keras
from keras import layers, models
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
```

### Mudanças no Carregamento de Imagens

**Antes (Depreciado):**
```python
from tensorflow.keras.preprocessing.image import load_img, img_to_array
img = load_img(path, target_size=(150, 150))
```

**Depois (Moderno):**
```python
img = keras.utils.load_img(path, target_size=(150, 150))
img_array = keras.utils.img_to_array(img)
```

### Mudanças no Data Augmentation

**Antes (Depreciado):**
```python
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    ...
)
train_generator = train_datagen.flow_from_directory(...)
```

**Depois (Moderno):**
```python
# Camadas de augmentation
data_augmentation = keras.Sequential([
    layers.RandomRotation(0.1),
    layers.RandomTranslation(0.2, 0.2),
    layers.RandomZoom(0.2),
    layers.RandomFlip("horizontal"),
])

# Carregamento com tf.data
train_dataset = keras.utils.image_dataset_from_directory(...)
train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
```

### Resumo das APIs Atualizadas

| Função Legada | Função Moderna |
|---------------|----------------|
| `tensorflow.keras.*` | `keras.*` |
| `ImageDataGenerator` | `keras.utils.image_dataset_from_directory()` + `tf.data` |
| `flow_from_directory()` | `keras.utils.image_dataset_from_directory()` |
| `load_img()` | `keras.utils.load_img()` |
| `img_to_array()` | `keras.utils.img_to_array()` |
| `model.save('file.h5')` | `model.save('file.keras')` |

---

## 📊 Comparativo de Complexidade

| Característica | Cães e Gatos | Tumores Cerebrais |
|----------------|--------------|-------------------|
| Número de classes | 2 | 4 |
| Complexidade do problema | Baixa | Alta |
| Importância de erros | Baixa | Alta (aplicação médica) |
| Necessidade de métricas | Básica | Detalhada |
| Regularização | Mínima | Extensiva (Dropout, BatchNorm) |

---

## 🎯 Conclusão

As principais adaptações foram necessárias para:

1. **Lidar com mais classes:** De 2 para 4 classes, exigindo mudança na camada de saída e função de perda
2. **Melhorar generalização:** Data augmentation e regularização para evitar overfitting
3. **Avaliação mais rigorosa:** Métricas detalhadas essenciais para aplicações médicas
4. **Eficiência no carregamento:** `tf.data` e `image_dataset_from_directory()` para datasets maiores
5. **Monitoramento do treinamento:** Callbacks para otimizar o processo de treinamento
6. **Atualização para APIs modernas:** Uso do Keras 3.x com `tf.data` em vez de APIs depreciadas

O notebook de tumores cerebrais representa uma evolução significativa em termos de boas práticas de deep learning, adequação para um problema real de diagnóstico médico e uso de APIs modernas e recomendadas do Keras/TensorFlow.

---

## 🖥️ Adaptação para VS Code (Execução Local)

O notebook foi adaptado para permitir execução tanto no Google Colab quanto localmente no VS Code. As seguintes mudanças foram realizadas:

### Célula 1 - Código do Colab Comentado

O código específico do Google Colab foi comentado e substituído por uma detecção de ambiente:

```python
# ============================================================
# CÓDIGO DO GOOGLE COLAB (COMENTADO)
# Descomente as linhas abaixo se estiver rodando no Google Colab
# ============================================================
# from google.colab import drive
# drive.mount('/content/drive')

# Detecção automática do ambiente (VS Code local)
IN_COLAB = False
print(f"Ambiente: {'Google Colab' if IN_COLAB else 'VS Code Local'}")
```

### Célula 3 - Configuração do Ambiente

O bloco de montagem do Google Drive e mudança de diretório foi comentado:

```python
# ============================================================
# CÓDIGO DO GOOGLE COLAB (COMENTADO)
# Descomente o bloco abaixo se estiver rodando no Google Colab
# ============================================================
# if IN_COLAB:
#     from google.colab import drive
#     drive.mount("/content/drive")
#     os.chdir("/content/drive/My Drive/Colab Notebooks/trabalho_final")

# Para VS Code: não é necessário mudar diretório
print(f"Rodando no ambiente: {'Google Colab' if IN_COLAB else 'VS Code Local'}")
print(f"Diretório atual: {os.getcwd()}")
```

### Célula 4 - Caminhos dos Diretórios (Simplificado)

Removidos os condicionais `if IN_COLAB` e usado `os.path.join()` para compatibilidade:

```python
# Usando os.path.join para compatibilidade entre sistemas operacionais
base_dir = os.path.join(".", "Brain Tumor MRI Dataset")
train_dir = os.path.join(base_dir, "Training")
test_dir = os.path.join(base_dir, "Testing")
```

### Célula 5 - Contagem de Imagens (Simplificado)

Removidos os condicionais e padronizado o uso de `os.path.join()`:

```python
for classe in classes:
    path = os.path.join(train_dir, classe)  # Compatível com qualquer SO
    count = len(os.listdir(path))
```

### Como Executar

**No VS Code (Local):**
1. Certifique-se de que o notebook está na pasta `trabalho_final`
2. A pasta `Brain Tumor MRI Dataset` deve estar no mesmo diretório do notebook
3. Execute as células normalmente usando o Jupyter no VS Code
4. Mantenha `IN_COLAB = False` na primeira célula

**No Google Colab:**
1. Altere `IN_COLAB = True` na primeira célula
2. Descomente as linhas de montagem do Google Drive
3. Descomente o bloco de configuração de ambiente na célula 3
4. Execute normalmente

### Vantagens da Adaptação

| Aspecto | Google Colab | VS Code Local |
|---------|--------------|---------------|
| **GPU** | Gratuita (limitada) | Depende do hardware |
| **Armazenamento** | Google Drive | Disco local |
| **Velocidade I/O** | Mais lento (rede) | Mais rápido (local) |
| **Sessão** | Expira após inatividade | Persistente |
| **Debugging** | Limitado | Completo |

---

## 🔧 Revisão de Código e Correções

O notebook passou por uma revisão completa para identificar problemas e aplicar melhorias. Abaixo estão documentadas todas as correções realizadas:

### ❌ Problemas Identificados e Corrigidos

#### 1. **Ordem das Células Incorreta**

**Problema:** A célula de carregamento do modelo salvo estava no final do notebook (após o treinamento e avaliação), quando deveria estar ANTES da célula de treinamento.

**Correção:** A célula foi movida para logo após a compilação do modelo, permitindo que o fluxo de execução detecte se existe um modelo salvo antes de decidir treinar.

**Localização:** Nova posição após a célula "Compilar o modelo"

---

#### 2. **Variável `TREINAR` Não Definida**

**Problema:** A variável `TREINAR` era usada na célula de treinamento, mas só era definida na célula de carregamento do modelo (que estava fora de ordem). Isso causaria erro `NameError: name 'TREINAR' is not defined`.

**Correção:** A célula de verificação do modelo salvo agora está posicionada corretamente e define `TREINAR = True` ou `TREINAR = False` baseado na existência do arquivo.

```python
if os.path.exists(modelo_salvo):
    model = keras.models.load_model(modelo_salvo)
    TREINAR = False
else:
    TREINAR = True
```

---

#### 3. **Erro ao Plotar Histórico sem Treinar**

**Problema:** Se o modelo fosse carregado de arquivo (sem treinar), a variável `history` não existiria, causando erro ao tentar plotar o histórico de treinamento.

**Correção:** Adicionada verificação condicional:

```python
if TREINAR and 'history' in dir():
    plot_training_history(history)
else:
    print("ℹ️ Histórico de treinamento não disponível (modelo carregado de arquivo).")
```

---

#### 4. **Import Duplicado**

**Problema:** O módulo `os` era importado novamente na célula de carregamento do modelo, sendo desnecessário pois já havia sido importado anteriormente.

**Correção:** Removido o import duplicado.

---

#### 5. **Função `predict_single_image` Melhorada**

**Problemas:**
- Não verificava se o arquivo de imagem existia
- Usava lista hardcoded de classes em vez da variável `class_names`
- Caminho hardcoded para imagem de teste

**Correções:**
```python
def predict_single_image(model, image_path, img_size=(150, 150)):
    # Verificar se o arquivo existe
    if not os.path.exists(image_path):
        print(f"❌ Erro: Arquivo não encontrado: {image_path}")
        return None, None
    
    # Usar class_names se disponível
    class_labels = class_names if 'class_names' in dir() else ['glioma', 'meningioma', 'notumor', 'pituitary']
    
    # ... resto do código
```

---

### ✅ Melhorias Aplicadas

#### 1. **Feedback Visual Aprimorado**

Adicionados emojis e mensagens mais claras para indicar o status:

```python
print(f"✅ Modelo salvo encontrado: {modelo_salvo}")
print(f"⚠️ Modelo não encontrado: {modelo_salvo}")
print("ℹ️ Histórico de treinamento não disponível")
```

---

#### 2. **Fluxo de Execução Otimizado**

O notebook agora segue a ordem correta:

1. Configuração do ambiente
2. Carregamento dos dados
3. Criação e compilação do modelo
4. **Verificação de modelo salvo** ← Nova posição
5. Treinamento (se necessário)
6. Avaliação
7. Salvamento

---

#### 3. **Exemplo de Uso Dinâmico**

A função `predict_single_image` agora tem um exemplo que usa o próprio dataset:

```python
# Para testar com uma imagem do dataset:
exemplo_imagem = os.path.join(test_dir, "glioma", os.listdir(os.path.join(test_dir, "glioma"))[0])
predicted_class, confidence = predict_single_image(model, exemplo_imagem)
```

---

### 📋 Checklist de Qualidade

| Item | Status |
|------|--------|
| Ordem das células correta | ✅ |
| Variáveis definidas antes do uso | ✅ |
| Tratamento de erros | ✅ |
| Imports sem duplicação | ✅ |
| Compatibilidade Windows/Linux | ✅ |
| Feedback ao usuário | ✅ |
| Código documentado | ✅ |
| APIs modernas do Keras | ✅ |

---

## 📚 Referências

- [Keras 3.x Documentation](https://keras.io/)
- [TensorFlow Data API](https://www.tensorflow.org/guide/data)
- [Image Classification Tutorial](https://www.tensorflow.org/tutorials/images/classification)
- [Data Augmentation Layers](https://keras.io/api/layers/preprocessing_layers/image_augmentation/)
- [Jupyter in VS Code](https://code.visualstudio.com/docs/datascience/jupyter-notebooks)
