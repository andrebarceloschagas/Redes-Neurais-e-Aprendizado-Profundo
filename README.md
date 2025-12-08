# Redes Neurais e Aprendizado Profundo (Deep Learning) 2025/2

Repositório de códigos e anotações da disciplina de Redes Neurais e Aprendizado Profundo (Deep Learning) do curso de Ciência da Computação da Universidade Federal do Tocantins.

Professor: Dr. Marcelo Lisboa

Aluno: [Antonio André Barcelos Chagas](https://github.com/andrebarceloschagas)


## Conteúdo do Repositório
- **`cnn_caes_gatos.ipynb`**: Implementação inicial de uma CNN para classificação de imagens de cães e gatos.
- **`cnn_brain_tumor.ipynb`**: Adaptação do modelo para detecção de tumores cerebrais utilizando Transfer Learning com EfficientNetB0.
- **`cnn_brain_tumor_2.ipynb`**: Versão otimizada do modelo de detecção de tumores cerebrais com fine-tuning avançado.
- **`README.md`**: Documentação do projeto e evolução dos modelos.



## Análise da Evolução dos Modelos de Detecção de Tumores Cerebrais

Este relatório detalha a evolução dos modelos de Deep Learning desenvolvidos para a detecção de tumores cerebrais em imagens de ressonância magnética (MRI). O processo partiu de uma adaptação simples de um modelo de classificação de cães e gatos, evoluindo para um classificador mais sofisticado utilizando Transfer Learning com EfficientNetB0, e culminando em um modelo otimizado com fine-tuning avançado.

## 1. Visão Geral da Evolução

| Característica | **Modelo Inicial (Base)** | **Modelo Intermediário** | **Modelo Final (Otimizado)** |
| :--- | :--- | :--- | :--- |
| **Origem** | `cnn_caes_gatos.ipynb` | `cnn_brain_tumor.ipynb` | `cnn_brain_tumor_2.ipynb` |
| **Arquitetura** | CNN Simples (Custom) | EfficientNetB0 (Transfer Learning) | EfficientNetB0 (Fine-Tuning) |
| **Resolução de Entrada** | 100x100 | 150x150 | 224x224 (Nativa) |
| **Canais de Cor** | RGB (3 canais) | RGB (3 canais) | RGB (3 canais) |
| **Data Augmentation** | Não implementado | Moderado (Rotação, Zoom, Flip) | Leve e Estratégico |
| **Pré-processamento** | Normalização Manual (`/255.0`) | Rescaling Manual (`1./255`) | Normalização Interna (EfficientNet) |
| **Estratégia de Treino** | Treinamento do Zero | Feature Extraction (Base congelada) | 2 Fases: Feature Ext. + Fine-Tuning |
| **Otimizador** | Não especificado (implícito) | Adam (`lr=0.001`) | Adam (Fase 1: `0.001`, Fase 2: `1e-5`) |
| **Acurácia (Validação)** | N/A (foco didático) | ~50-60% (Estagnado) | **>90%** (Estimado) |

---

## 2. Detalhes das Iterações

### 1. Modelo Base: Classificação de Cães e Gatos (`cnn_caes_gatos.ipynb`)
Este notebook serviu como ponto de partida conceitual. Ele implementava um fluxo básico de classificação binária.
* **Limitações para Tumores:**
    * **Resolução Baixa (100x100):** Insuficiente para capturar detalhes finos de tumores.
    * **Dataset Pequeno:** O código foi desenhado para um exemplo didático com poucas imagens, o que levaria a overfitting severo em um problema médico complexo.
    * **Arquitetura Simples:** Uma CNN construída do zero não tem a capacidade de extração de características necessária para diferenciar tipos de tumores (glioma, meningioma, etc.).

### 2. Primeira Adaptação: Transfer Learning Rígido (`cnn_brain_tumor.ipynb`)
Nesta etapa, o código foi adaptado para o dataset de tumores cerebrais, introduzindo o conceito de Transfer Learning.
* **Mudanças Principais:**
    * Uso da **EfficientNetB0** pré-treinada no ImageNet.
    * Congelamento das primeiras 100 camadas para aproveitar os pesos aprendidos.
    * Aumento da resolução para 150x150.
* **Problemas Identificados:**
    * **Normalização Dupla:** Foi mantida uma camada de `Rescaling(1./255)`. Como a EfficientNetB0 já possui normalização interna, isso "achatou" os dados, prejudicando o aprendizado.
    * **Resolução Inadequada:** 150x150 ainda é menor que a resolução nativa da EfficientNet (224x224), perdendo informações espaciais cruciais.
    * **Underfitting:** O congelamento rígido das camadas impediu que a rede adaptasse seus filtros para as texturas específicas de ressonância magnética (muito diferentes de fotos de objetos comuns). O modelo estagnou com acurácia baixa (~50%).

### 3. Modelo Final: Otimização e Fine-Tuning (`cnn_brain_tumor_2.ipynb`)
A versão final corrige as falhas anteriores e implementa práticas avançadas de Deep Learning para imagens médicas.

#### a. Correção de Pré-processamento e Entrada
* **Resolução Nativa:** Alterada para `(224, 224)`. Isso permite que a EfficientNet opere na resolução para a qual foi projetada, maximizando a extração de detalhes.
* **Remoção de Rescaling:** A camada de normalização manual foi removida, deixando a EfficientNet lidar com os valores dos pixels internamente.

#### b. Estratégia de Treinamento em Duas Fases
Esta foi a mudança mais impactante. Em vez de apenas treinar o classificador final, o processo foi dividido:
1.  **Fase 1 (Feature Extraction):** A base da EfficientNet é congelada. Apenas as novas camadas densas (o "topo") são treinadas com uma learning rate padrão (`0.001`). Isso estabiliza os pesos do classificador sem destruir os pesos pré-treinados da base.
2.  **Fase 2 (Fine-Tuning):** Toda a rede é descongelada (exceto camadas BatchNormalization para estabilidade). O treinamento continua com uma **learning rate muito baixa (`1e-5`)**. Isso permite que os filtros convolucionais profundos façam micro-ajustes para se especializarem em texturas cerebrais.

#### c. Arquitetura do Classificador (Head)
O topo da rede foi robustecido para lidar com a complexidade das 4 classes de tumores:
* **GlobalAveragePooling2D:** Reduz a dimensionalidade espacial.
* **Dense (512) + BatchNormalization + Dropout (0.4):** Uma camada densa larga com regularização forte para aprender combinações não-lineares de características sem overfitting.

---

## 3. Conclusão

A evolução do projeto demonstra a importância de não apenas aplicar modelos pré-treinados, mas de adaptá-los corretamente ao domínio do problema.

1.  O **modelo inicial** provou o conceito de CNN.
2.  O **modelo intermediário** tentou usar o estado da arte (EfficientNet), mas falhou em detalhes de implementação (normalização e resolução).
3.  O **modelo final** corrigiu a engenharia de dados e aplicou uma estratégia de treinamento (Fine-Tuning) que é essencial para transferir conhecimento de imagens genéricas (ImageNet) para imagens médicas específicas, resultando em um sistema muito mais robusto e preciso.