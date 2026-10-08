# Fix: exportação em modo `multiplo` incompatível com `InstanceParser`

## Resumo

`exportar_cenario(..., formato='multiplo')` gerava arquivos que `InstanceParser._parse_directory` não reconhecia. Exportar e reler resultava em instância vazia (sem cidades, sem itens, parâmetros padrão).

## Sintoma

`InstanceParser.carregar(dir)` retorna `TTPInstance` com `grafo.cidades` vazio, `itens=[]` e parâmetros nos valores padrão. Nenhum erro é levantado: o parser usa `os.path.exists` e ignora arquivos ausentes, então a falha é silenciosa.

## Causa

Três divergências entre exportador e parser:

1. **Nomes de arquivo diferentes.**
2. **Itens em um arquivo só no exportador.** O parser espera dois: definição (`id profit weight`) e distribuição (`id node`).
3. **Distâncias EXPLICIT em arquivo separado.** O exportador gravava `distances.txt`. O parser lê trios `u v dist` de dentro de `mapa_cidade.txt` e ignora `distances.txt`.

## Mapeamento de arquivos

| Antes (exportador) | Depois | Conteúdo |
|---|---|---|
| `params.txt` | `parametros_globais.txt` | Cabeçalho `CHAVE: valor` |
| `nodes.txt` + `distances.txt` | `mapa_cidade.txt` | `id x y` (EUC_2D) ou `u v dist` (EXPLICIT) |
| `items.txt` | `definicao_objetos.txt` | `id profit weight` |
| (não existia) | `distribuicao_objetos.txt` | `id node` |

## Mudança

Função `_export_multiple_files`:

```python
def _export_multiple_files(instance: TTPInstance, base_dir: Path) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)

    # Params
    with open(base_dir / "parametros_globais.txt", 'w', encoding='utf-8') as f:
        f.write(f"PROBLEM NAME: {instance.nome}\n")
        f.write(f"DIMENSION: {instance.dimension}\n")
        f.write(f"NUMBER OF ITEMS: {instance.num_items}\n")
        f.write(f"CAPACITY OF KNAPSACK: {instance.capacidade_mochila}\n")
        f.write(f"MIN SPEED: {instance.v_min}\n")
        f.write(f"MAX SPEED: {instance.v_max}\n")
        f.write(f"RENTING RATIO: {instance.taxa_aluguel}\n")
        f.write(f"EDGE_WEIGHT_TYPE: {instance.grafo.edge_weight_type}\n")
        f.write(f"DEGRADATION CONSTANT: {instance.constante_degradacao}\n")

    # Map: coords (EUC_2D) or distance triplets (EXPLICIT)
    with open(base_dir / "mapa_cidade.txt", 'w', encoding='utf-8') as f:
        if instance.grafo.edge_weight_type == 'EUC_2D':
            for cid, cidade in instance.grafo.cidades.items():
                f.write(f"{cid}\t{cidade.x:.4f}\t{cidade.y:.4f}\n")
        else:
            for (n1, n2), dist in instance.grafo.distancias.items():
                f.write(f"{n1}\t{n2}\t{dist:.4f}\n")

    # Item definitions
    with open(base_dir / "definicao_objetos.txt", 'w', encoding='utf-8') as f:
        for item in instance.itens:
            f.write(f"{item.id}\t{item.valor:.4f}\t{item.peso:.4f}\n")

    # Item distribution
    with open(base_dir / "distribuicao_objetos.txt", 'w', encoding='utf-8') as f:
        for item in instance.itens:
            f.write(f"{item.id}\t{item.cidade_origem}\n")
```

## Notas de comportamento

- `distances.txt` deixou de existir. O parser nunca o lia.
- Em EXPLICIT, o parser cria cidades ausentes via `range(1, dimension + 1)`. `mapa_cidade.txt` não tem coordenadas, então `x` e `y` voltam nos valores padrão de `Cidade`.
- O parser só cria um `Item` se o `id` existir em **ambos** `definicao_objetos.txt` e `distribuicao_objetos.txt`. O exportador grava os mesmos ids nos dois, então a condição é satisfeita.

## Teste

```python
def test_exportar_cenario_multiplo_and_parse(tmp_path):
    g = gerar_grafo_normal(n_cidades=4, seed=42)
    itens = gerar_itens(n_itens=4, cidades=g.cidades, seed=42)
    cap = calcular_capacidade_mochila(itens, tightness_ratio=0.6)

    inst = TTPInstance(
        nome='TestExport',
        grafo=g,
        itens=itens,
        capacidade_mochila=cap,
        v_min=0.1,
        v_max=1.0,
        taxa_aluguel=1.5,
        constante_degradacao=10.0,
        dimension=4,
        num_items=4,
    )

    exportar_cenario(inst, str(tmp_path / "inst"), formato='multiplo')
    parsed = InstanceParser.carregar(str(tmp_path / "inst"))

    assert parsed.nome == 'TestExport'
    assert len(parsed.grafo.cidades) == 4
    assert len(parsed.itens) == 4
    assert parsed.constante_degradacao == 10.0
```

`tmp_path` é fixture do pytest e remove a pasta temporária sozinho.