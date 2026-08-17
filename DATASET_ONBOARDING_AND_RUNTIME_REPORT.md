# Guia de integração de datasets e execução do HBGL

Este documento registra o trabalho realizado para integrar o Eurlex-4K ao
HBGL e transforma as decisões, problemas e medições da execução em um roteiro
reutilizável para outros datasets e máquinas semelhantes.

O objetivo não é apenas mostrar comandos. Antes de iniciar um experimento de
vários dias, é necessário validar quatro contratos independentes:

1. o contrato dos dados e da taxonomia;
2. o contrato de identidade usado na avaliação;
3. o contrato de treino, incluindo comprimento, batch e inicialização;
4. o contrato da máquina realmente disponível dentro do container.

Copiar somente os parâmetros de outro dataset pode produzir um experimento
lento, impossível de avaliar ou metodologicamente inconsistente.

## 1. Visão geral do HBGL

Nesta implementação, o HBGL trata classificação hierárquica como geração
seq2seq. Cada nível da hierarquia ocupa uma posição-alvo e cada rótulo recebe
um token estável no formato `[A_<id>]`.

Os datasets canônicos são mantidos somente para leitura. O adaptador converte
cada fold para artefatos compatíveis com o HBGL, sem modificar os PKLs de
origem. Os arquivos preparados ficam em:

```text
resource/prepared-datasets/<DATASET>/fold_<N>/
```

Os principais artefatos são:

- `train.jsonl`, `val.jsonl` e `test.jsonl`;
- `label_map.pkl`, com nomes canônicos e tokens estáveis;
- `label_taxonomy.tsv`, na representação esperada pelo HBGL;
- sidecars `<split>_document_ids.json`, com a identidade de avaliação;
- `manifest.json`, com fingerprint, versão, contagens, cobertura e
  sobreposições.

O treinamento gera checkpoints e predições em:

```text
models/<RUN_NAME>/
```

Cada fold deve usar um `RUN_NAME` próprio. Não compartilhe diretórios de
saída entre folds ou configurações diferentes.

## 2. O que foi necessário para integrar o Eurlex-4K

### 2.1 Dados canônicos observados

O Eurlex-4K fornecido possui:

- 19.314 linhas em `samples.pkl`;
- 19.281 documentos externos únicos;
- 3.956 rótulos canônicos;
- cinco folds originais;
- uma taxonomia plana em `label_taxonomy.pkl`;
- identidade externa `text_idx`;
- qrels canônicos em `relevance_map.pkl`;
- classes head/tail em `label_cls.pkl` e `text_cls.pkl`.

Os folds são listas de posições de `samples.pkl`. Portanto, `idx` e
`text_idx` têm funções diferentes:

- `idx` localiza uma linha no PKL;
- `text_idx` identifica um documento externo na avaliação.

Nunca use `text_idx` como índice posicional de `samples.pkl`.

### 2.2 Taxonomia plana

A taxonomia foi convertida para:

```text
Root -> nome_do_rotulo_0, nome_do_rotulo_1, ..., nome_do_rotulo_3955
```

Os filhos são ordenados pelo ID canônico. A conversão para nomes é necessária
porque o HBGL usa os nomes para construir a taxonomia textual e para a
inicialização semântica dos tokens de rótulo. Os IDs continuam preservados nos
tokens `[A_<id>]`, o que mantém a relação com os artefatos canônicos e torna os
checkpoints reproduzíveis entre folds.

O adaptador valida que:

- existe somente a raiz esperada;
- a raiz contém IDs inteiros sem duplicatas;
- seus filhos coincidem exatamente com os IDs observados;
- cada ID possui um único nome e cada nome possui um único ID;
- todos os mapas de avaliação cobrem documentos e rótulos válidos.

Como a hierarquia possui um nível de rótulos, o alvo precisa de uma posição de
rótulos e uma posição EOS/SEP:

```text
MAX_TARGET_LENGTH=2
```

Essa regra não deve ser copiada para um dataset hierárquico. Para um novo
dataset, derive o comprimento da profundidade efetivamente exportada e inclua
a posição terminal exigida pelo decoder. Confirme o resultado com um smoke
test.

### 2.3 Identidade, duplicatas e gold

O Eurlex possui mais linhas que documentos externos. A política adotada foi:

- preservar as posições originais dos folds;
- preservar documentos repetidos que atravessam train/val/test;
- registrar sobreposições no manifesto;
- no teste, conservar a primeira ocorrência de cada `text_idx` na ordem do
  fold;
- avaliar cada documento externo uma única vez;
- usar `relevance_map.pkl` como gold canônico.

Isso é importante porque duas linhas repetidas podem não carregar exatamente
a mesma anotação local. A avaliação não deve depender de qual duplicata foi
encontrada primeiro para obter o gold.

No fold 1, por exemplo:

| Medida | Train | Val | Test |
|---|---:|---:|---:|
| Linhas do fold | 13.905 | 1.546 | 3.863 |
| Documentos externos únicos | 13.885 | 1.545 | 3.863 |
| Rótulos presentes nas linhas | 3.701 | 1.869 | 2.686 |

O manifesto também registrou três IDs compartilhados entre train/val, nove
entre train/test e nenhum entre val/test. Essas sobreposições fazem parte dos
folds fornecidos e não foram removidas.

### 2.4 Candidatos não vistos no treino

O ranking é denso somente sobre os rótulos suportados pelo treino do fold:

| Fold | Rótulos suportados | Documentos externos no teste |
|---:|---:|---:|
| 0 | 3.705 | 3.863 |
| 1 | 3.701 | 3.863 |
| 2 | 3.681 | 3.863 |
| 3 | 3.741 | 3.862 |
| 4 | 3.713 | 3.859 |

Um rótulo canônico ausente do treino não pode ser pontuado pelo mask
fold-local do HBGL. Quando aparece no gold de teste, ele é contado como erro;
não é removido do denominador.

No fold 1, 255 dos 3.956 rótulos não foram vistos no treino. Portanto, a
cobertura de candidatos foi aproximadamente 93,6%. Essa informação deve
acompanhar qualquer comparação de métricas.

### 2.5 Fingerprint e cache

O fingerprint dos dados preparados inclui os folds e os artefatos canônicos
relevantes, incluindo taxonomia e mapas de avaliação. O manifesto do Eurlex
está atualmente na versão de artefato 5.

Se a fonte ou o contrato mudar, a preparação deve falhar em vez de reutilizar
silenciosamente um cache incompatível. Somente então use `--force` ou
`FORCE_PREPARE=1`, depois de confirmar que a substituição é intencional.

Não confunda:

- artefatos preparados, compartilháveis por fold e fingerprint;
- features tokenizadas de treino/val/test, armazenadas no diretório da run;
- checkpoints do modelo, também armazenados no diretório da run.

Uma nova run pode gastar vários minutos recriando features mesmo quando o
JSONL preparado já existe.

## 3. Configuração de treino usada no Eurlex

A configuração executável está em `configs/eurlex-4k.yaml`. No experimento do
fold 1 foram usados:

```yaml
PER_GPU_TRAIN_BATCH_SIZE: 16
NUM_TRAINING_STEPS: 96000
SAVE_STEPS: 3000
OMP_NUM_THREADS: 1
MKL_NUM_THREADS: 1
OPENBLAS_NUM_THREADS: 1
NUMEXPR_NUM_THREADS: 1
```

Além disso:

- `MAX_SOURCE_LENGTH=510`;
- `MAX_TARGET_LENGTH=2`;
- learning rate `3e-5`;
- warmup de 500 passos;
- gradient accumulation igual a 1;
- seed 42;
- `soft_label` e `soft_label_hier_real` habilitados;
- label smoothing igual a 0;
- label-CPT desabilitado;
- inicialização aleatória de rótulos desabilitada;
- rankings HGCLR em K=1,5,10 habilitados.

### Divergência de documentação sobre batch

O README ainda menciona batch 8 como default conservador do Eurlex. Entretanto,
o YAML atual e o fold 1 concluído usaram batch 16. Para comparação entre folds,
os folds restantes devem usar batch 16, ou todos os folds devem ser refeitos
sob um novo protocolo com batch 8.

Variáveis exportadas no shell têm precedência sobre o YAML. Antes de uma run,
verifique overrides antigos:

```bash
env | rg 'PER_GPU|TRAINING_STEPS|SAVE_STEPS|OMP|MKL|OPENBLAS|NUMEXPR'
```

Use `unset VARIAVEL` quando quiser voltar ao valor do YAML.

### 3.1 Por que o label-CPT foi desabilitado

O Eurlex possui 3.956 rótulos irmãos sob uma única raiz. O label-CPT aplicaria
atenção quadrática sobre esse conjunto, com custo de tempo e memória alto. A
taxonomia plana também não fornece relações internas úteis para justificar
esse custo.

Foi mantida a inicialização semântica existente: a representação inicial de
cada rótulo é a média dos embeddings WordPiece de seu nome. Para outro dataset,
decida label-CPT com base no número de rótulos, profundidade e estrutura real;
não o habilite apenas porque outro experimento o utilizou.

## 4. Conferir a máquina real antes de treinar

Em ambientes containerizados, `lscpu` e `nproc` podem mostrar os processadores
do host, não a cota efetivamente concedida ao container.

Na máquina usada neste trabalho:

- a oferta informava 8 vCPUs AMD EPYC 7543;
- `nproc` e o cpuset mostravam 64 CPUs lógicas visíveis;
- o cgroup limitava o container a `680000/100000`, ou 6,8 CPUs equivalentes;
- aproximadamente 90% dos períodos do cgroup apresentavam throttling durante
  o problema inicial;
- o processo Python chegou a criar cerca de 69 threads;
- a GPU ficava frequentemente ociosa esperando preparação em CPU.

Portanto, a quantidade relevante é a menor entre afinidade/cpuset e quota do
cgroup, não o número anunciado pelo host.

### 4.1 Checklist de CPU

```bash
nproc
lscpu
grep -E 'Cpus_allowed|Cpus_allowed_list' /proc/self/status
```

Para cgroup v2:

```bash
cat /sys/fs/cgroup/cpu.max
cat /sys/fs/cgroup/cpu.stat
```

Em `cpu.max`, `max 100000` significa sem quota. Por exemplo, `680000 100000`
equivale a 6,8 CPUs.

Para cgroup v1, os caminhos variam por imagem:

```bash
cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us
cat /sys/fs/cgroup/cpu/cpu.cfs_period_us
cat /sys/fs/cgroup/cpu/cpu.stat
```

Também pode existir `cpu,cpuacct` no caminho. Calcule:

```text
CPUs equivalentes = cpu.cfs_quota_us / cpu.cfs_period_us
```

Em `cpu.stat`, uma razão alta entre `nr_throttled` e `nr_periods` indica que
threads demais estão disputando uma cota pequena.

### 4.2 Checklist de GPU

```bash
nvidia-smi
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu,utilization.memory --format=csv
watch -n 2 nvidia-smi
```

Durante treino estável, acompanhe GPU, CPU e número de threads:

```bash
pgrep -af '/workspace/flaviossf/HBGL/run.py'
ps -o pid,etime,pcpu,pmem,nlwp,stat,cmd -p <PID>
ps -L -p <PID> | wc -l
```

GPU baixa acompanhada de CPU no limite e throttling alto normalmente significa
starvation da GPU por pré-processamento, collation ou bibliotecas BLAS.

### 4.3 Oversubscription encontrada e correção

O pipeline de soft labels constrói alvos densos. Em um microbenchmark desta
máquina, a construção de um batch levou aproximadamente:

- 0,105 s com uma thread;
- 1,007 s com oito threads;
- mais de 25 s quando testada com a configuração excessiva de threads.

Mais threads foram muito piores por causa da cota do cgroup, sincronização e
contenção. O treino lento chegou a aproximadamente 16,36 s/passo, com GPU
frequentemente ociosa.

A correção foi limitar explicitamente os pools independentes:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

Após a correção, a run longa apresentou intervalos de aproximadamente 14 min
45 s entre validações separadas por 3.000 updates, cerca de 0,3 s/update
incluindo o custo amortizado da validação. Isso é uma referência desta máquina,
não uma garantia para outra GPU ou imagem.

Comece com uma thread em containers com quota pequena. Depois faça benchmarks
controlados com 1, 2 e 4 threads. Não use automaticamente o valor mostrado por
`nproc`.

## 5. Problemas de código encontrados

### 5.1 Avaliador com matriz densa inútil

O caminho de avaliação principal criava uma matriz de confusão 3.956 x 3.956
e executava atualizações aninhadas que não eram usadas nas métricas retornadas.
Com rankings densos, isso fazia a validação parecer travada em CPU.

A alocação e as atualizações inúteis foram removidas do caminho `evaluate`
usado pela execução. As contagens necessárias para precision, recall e F1 foram
preservadas.

Ainda existe uma alocação semelhante, não utilizada, em `evaluate_seq2seq`.
Ela não bloqueou o caminho usado nesta run, mas deve ser removida antes de usar
essa função com milhares de rótulos.

### 5.2 Perda SEP usando tensores errados

No ramo `soft_label_hier_real`, o código construía tensores reduzidos para SEP:

```python
prediction_scores_masked_sep
label_ids_sep
```

mas chamava a BCE com os tensores completos. A chamada foi corrigida para:

```python
pseudo_lm_loss += loss_fct(
    prediction_scores_masked_sep,
    label_ids_sep,
)
```

Isso corrige a função objetivo e evita trabalho desnecessário nesse componente.
O ganho isolado é menor que o obtido com a correção de threads, porque a cabeça
do modelo ainda calcula logits sobre o vocabulário completo.

### Ressalva de reprodutibilidade do fold 1

O fold 1 foi retomado do `ckpt-9000`, que havia sido treinado antes da correção
SEP. Os passos posteriores usaram o código corrigido. Assim, esse fold contém
9.000 passos com a função antiga e o restante com a função corrigida.

Para um experimento de cinco folds metodologicamente rigoroso, refaça o fold 1
do zero com o código corrigido. Não misture suas métricas com folds inteiramente
treinados após a correção sem registrar essa diferença.

## 6. Procedimento recomendado para um novo dataset

### Etapa 1: inspecionar os dados sem modificá-los

Registre:

- quantidade de linhas e documentos únicos;
- campos e tipos de cada amostra;
- identidade posicional e identidade externa;
- quantidade e formato dos rótulos;
- profundidade máxima e presença de ciclos/múltiplos pais;
- estrutura dos folds;
- duplicatas dentro e entre splits;
- fonte canônica do gold;
- mapas head/tail e frequências, quando existirem.

Arquivos pickle só devem ser abertos quando sua origem for confiável.

### Etapa 2: definir o contrato de identidade

Escolha explicitamente:

- como os folds apontam para amostras;
- qual ID aparece nos rankings;
- qual ID indexa qrels;
- como duplicatas são colapsadas;
- se sobreposições entre splits são preservadas ou constituem erro de dados.

Não adote a política do Eurlex sem verificar o novo dataset.

### Etapa 3: construir e validar a taxonomia

Para cada rótulo, valide nome, ID, pai, profundidade e alcançabilidade a partir
da raiz. Defina uma ordenação determinística. Para hierarquias reconstruídas a
partir de amostras, assegure que todas as relações observadas sejam
consistentes. Quando houver uma taxonomia externa, prefira-a a inferências a
partir da ordem dos rótulos de uma amostra.

### Etapa 4: produzir tokens estáveis

Use IDs canônicos para gerar `[A_<id>]`. Evite enumerar rótulos na ordem em que
aparecem no fold, pois isso altera o significado dos tokens entre folds.

### Etapa 5: preparar artefatos por fold

O adaptador deve gerar JSONL, mapas, taxonomia, sidecars e manifesto de forma
atômica. Inclua todos os arquivos canônicos relevantes no fingerprint e
incremente a versão do artefato quando o formato ou a semântica mudar.

### Etapa 6: configurar o modelo

Defina por dataset:

- `MAX_SOURCE_LENGTH`;
- `MAX_TARGET_LENGTH`;
- batch por GPU;
- gradient accumulation e batch efetivo;
- quantidade total de passos e warmup;
- label-CPT;
- inicialização semântica ou aleatória;
- política de candidatos não vistos;
- intervalos de logging, validação e checkpoint.

Não mude batch, acumulação, precisão ou função objetivo no meio de um fold se
o objetivo for uma comparação controlada.

### Etapa 7: adaptar a avaliação

Garanta que rankings, qrels, frequências e classes usam o mesmo namespace de
IDs. Valide a cobertura documental antes de calcular métricas. Registre:

- documentos esperados e produzidos;
- candidatos suportados;
- rótulos gold não pontuáveis;
- protocolo e parâmetros de propensão;
- K avaliados;
- política head/tail.

### Etapa 8: testes

Antes da GPU, crie testes para:

- taxonomia válida e malformada;
- bijeção ID/nome;
- qrels com documentos ou rótulos desconhecidos;
- duplicatas e sobreposições;
- fingerprint e rejeição de cache incompatível;
- JSONL e sidecars preparados;
- comprimento de entrada e alvo;
- candidatos suportados e gold não visto;
- cobertura de documentos no ranking;
- regressão dos datasets já suportados.

Execute a suíte local:

```bash
python -m pytest -q
```

## 7. Fluxo de execução

### 7.1 Validar a fonte

Para um dataset já implementado:

```bash
python dataset_adapter.py validate \
  --dataset-dir /caminho/para/DATASET \
  --dataset-name DATASET
```

No Eurlex:

```bash
python dataset_adapter.py validate \
  --dataset-dir /workspace/flaviossf/datasets/Eurlex-4k \
  --dataset-name Eurlex-4k
```

A validação atual retorna 19.314 amostras, 3.956 rótulos e os cinco folds.

### 7.2 Preparar um fold

```bash
python dataset_adapter.py prepare \
  --dataset-dir /workspace/flaviossf/datasets/Eurlex-4k \
  --dataset-name Eurlex-4k \
  --fold 0 \
  --prepared-data-dir resource/prepared-datasets
```

Inspecione o `manifest.json` gerado antes de treinar.

### 7.3 Smoke test

Use um diretório novo e apenas 12 passos:

```bash
NUM_TRAINING_STEPS=12 \
SAVE_STEPS=6 \
bash run_eurlex.sh 0 eurlex-smoke-fold-0
```

Critérios mínimos:

- treinamento chega ao passo 12 sem NaN ou erro de shape;
- checkpoint é salvo e recuperável;
- validação e teste terminam;
- ranking contém o número esperado de documentos;
- metadata contém o número esperado de candidatos;
- métricas são escritas;
- GPU é usada durante forward/backward;
- CPU não permanece throttled por oversubscription.

Não use a qualidade de um smoke de 12 passos para avaliar o modelo.

O smoke completo pode ser dominado por tokenização e avaliação. Nesta
integração, uma run nova gastou cerca de nove minutos preparando as features e
cerca de dois minutos para avaliar cada ranking denso, enquanto os 12 updates
foram uma fração pequena do tempo total.

### 7.4 Run completa

```bash
export PATH=/workspace/flaviossf/miniconda3/envs/hbgl/bin:$PATH
export PYTHON_BIN=/workspace/flaviossf/miniconda3/envs/hbgl/bin/python
export PER_GPU_TRAIN_BATCH_SIZE=16
export NUM_TRAINING_STEPS=96000
export SAVE_STEPS=3000
export EXPORT_RANKINGS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

bash run_eurlex.sh 0 eurlex-fold-0-batch16
```

Com uma GPU, execute folds sequencialmente:

```bash
for FOLD in 0 1 2 3 4; do
  bash run_eurlex.sh "$FOLD" "eurlex-fold-${FOLD}-batch16"
done
```

Use `tmux` ou outro supervisor para não perder o processo ao desconectar, mas
mantenha logs e nomes de run separados.

### 7.5 Retomada

`run_fold.sh` recusa um diretório de saída existente para evitar sobrescrita.
Para retomar, execute `run.py` diretamente com exatamente os mesmos argumentos
e o mesmo `output_dir`.

`run.py` procura o maior `ckpt-N` que contenha simultaneamente:

- `pytorch_model.bin`;
- `optimizer.bin`.

Ele recupera modelo, optimizer, scheduler e `global_step`. Passos executados
depois do último checkpoint completo são perdidos.

Não aponte `model_name_or_path` para o checkpoint ao retomar. Mantenha o modelo
base original e o mesmo `output_dir`; a recuperação é feita pelo estado salvo
nesse diretório.

### 7.6 Retenção de checkpoints

Após cada validação, o código mantém checkpoints que melhoram macro-F1 ou
micro-F1 e remove os demais. Por isso, `ckpt-96000.valid` pode existir sem um
diretório completo `ckpt-96000`: o checkpoint final foi avaliado e removido por
não superar o melhor anterior.

No fold 1 observado, o checkpoint preservado e usado no teste foi
`ckpt-72000`. Tanto `best_micro` quanto `best_macro` apontaram para o mesmo
modelo.

## 8. Avaliação e interpretação

Os rankings Eurlex são densos sobre os candidatos suportados pelo fold e usam
o score sigmoid da Eq. 10 na profundidade do rótulo. Cada ranking do fold 1
ocupou aproximadamente 311 MB.

O avaliador reproduz o protocolo HGCLR com:

- `precision@K`;
- `ndcg@K`;
- `psprecision@K`;
- `psnDCG@K`;
- `Mac-F1@K`;
- `Mic-F1@K`;
- K igual a 1, 5 e 10;
- parâmetros de propensão A=0,55 e B=1,5;
- corpus Eurlex de 19.281 documentos externos únicos.

As métricas JSON são percentuais já arredondados. No fold 1, por exemplo,
head precision@1 foi 83,1 e tail precision@1 foi 56,6. Isso indica uma lacuna
considerável de cauda longa, mas um fold isolado não é resultado final.

Relate média e desvio padrão dos cinco folds. Inclua junto às métricas:

- candidatos suportados por fold;
- documentos avaliados por fold;
- código/configuração usados;
- política para rótulos não vistos;
- qualquer fold retomado com código diferente.

## 9. Oportunidades de desempenho futuras

As seguintes otimizações devem ser avaliadas separadamente, sempre com smoke e
comparação de métricas:

1. remover a matriz de confusão não utilizada de `evaluate_seq2seq`;
2. evitar ordenar os ~4 mil candidatos repetidamente para cada K no avaliador;
3. armazenar apenas top-K quando um ranking denso não for requisito do
   protocolo;
4. reutilizar features tokenizadas de forma versionada entre runs equivalentes;
5. calcular logits somente para SEP e tokens de rótulos nas posições-alvo;
6. testar AMP/fp16 em uma run nova, sem ativá-lo no meio de um checkpoint;
7. perfilar separadamente data loader, forward, loss, backward, validação e
   exportação.

A projeção restrita de logits pode trazer ganho relevante para datasets com
milhares de rótulos, mas é uma mudança arquitetural e exige regressão completa
de WOS, RCV1 e Eurlex.

## 10. Checklist antes de iniciar vários folds

- [ ] Fonte validada e mantida somente para leitura.
- [ ] IDs posicionais e externos diferenciados.
- [ ] Taxonomia validada, determinística e sem rótulos órfãos.
- [ ] Gold canônico e política de duplicatas definidos.
- [ ] Manifesto e fingerprint inspecionados.
- [ ] Comprimentos de source/target justificados.
- [ ] Batch efetivo e learning rate registrados.
- [ ] Label-CPT e inicialização de rótulos decididos explicitamente.
- [ ] Cota de CPU do cgroup verificada.
- [ ] Número de threads limitado e medido.
- [ ] GPU e memória verificadas.
- [ ] Smoke test completo aprovado.
- [ ] Checkpoint, teste e ranking confirmados.
- [ ] Todos os folds usam o mesmo código e configuração.
- [ ] Estratégia de retomada conhecida.
- [ ] Média/desvio dos folds planejados.

## 11. Arquivos relevantes desta implementação

- `dataset_adapter.py`: validação e preparação canônica;
- `configs/eurlex-4k.yaml`: configuração executável do Eurlex;
- `run_eurlex.sh`: carregamento do YAML e entrada canônica;
- `run_fold.sh`: montagem dos argumentos por dataset;
- `run.py`: treino, recuperação, validação e retenção de checkpoints;
- `s2s_ft/modeling.py`: função objetivo e correção SEP;
- `test.py`: inferência e exportação de rankings;
- `hbgl_ranking.py`: construção e métricas do ranking HBGL;
- `evaluate_hbgl_ranking.py`: CLI do protocolo HGCLR;
- `eval.py`: métricas legadas e caminhos de avaliação;
- `tests/`: regressões do adaptador, launcher e avaliador.

Este relatório deve ser atualizado sempre que o formato dos artefatos, a
função objetivo, a política de avaliação ou a configuração efetivamente usada
em produção mudar.
