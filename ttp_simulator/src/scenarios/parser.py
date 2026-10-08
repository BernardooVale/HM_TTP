import os
from typing import Optional
from src.core.models import TTPInstance, Grafo, Cidade, Item

class InstanceParser:

    @classmethod
    def carregar(cls, path: str) -> TTPInstance:
        return cls().parse(path)

    def parse(self, path: str) -> TTPInstance:
        """Auto-detects whether path is a single file or a directory and parses it."""
        if os.path.isdir(path):
            return self._parse_directory(path)
        else:
            return self._parse_file(path)

    def _parse_file(self, file_path: str) -> TTPInstance:
        params = {}
        cidades = {}
        itens = []
        distancias = {}

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        i = 0
        # Parse headers
        while i < len(lines):
            line = lines[i]
            if (line.startswith('NODE_COORD_SECTION') or 
                line.startswith('EDGE_WEIGHT_SECTION') or 
                line.startswith('ITEMS_SECTION')):
                break

            if ':' in line:
                key, val = [p.strip() for p in line.split(':', 1)]
                params[key.upper()] = val
            i += 1

        edge_weight_type = params.get('EDGE_WEIGHT_TYPE', 'EUC_2D').upper()

        dimension = int(params.get('DIMENSION', 0))
        num_items = int(params.get('NUMBER OF ITEMS', 0))
        capacidade = float(params.get('CAPACITY OF KNAPSACK', params.get('CAPACIDADE_MOCHILA', 0.0)))
        v_min = float(params.get('MIN SPEED', params.get('V_MIN', 0.1)))
        v_max = float(params.get('MAX SPEED', params.get('V_MAX', 1.0)))
        taxa = float(params.get('RENTING RATIO', params.get('TAXA_ALUGUEL_R', 1.0)))
        
        # Degradation constant support for DEGRADATION CONSTANT, DEGRADATION, CONSTANTE_DEGRADACAO_C
        degradacao_val = (
            params.get('DEGRADATION CONSTANT') or 
            params.get('DEGRADATION') or 
            params.get('CONSTANTE_DEGRADACAO_C') or 
            '1.0'
        )
        degradacao = float(degradacao_val)
        nome = params.get('PROBLEM NAME', os.path.splitext(os.path.basename(file_path))[0])

        # Parse sections
        while i < len(lines):
            line = lines[i]

            if line.startswith('NODE_COORD_SECTION'):
                i += 1
                while i < len(lines) and not (lines[i].startswith('ITEMS_SECTION') or lines[i].startswith('EDGE_WEIGHT_SECTION')):
                    parts = lines[i].split()
                    if len(parts) >= 3 and parts[0].isdigit():
                        c_id = int(parts[0])
                        x = float(parts[1])
                        y = float(parts[2])
                        cidades[c_id] = Cidade(id=c_id, x=x, y=y)
                    i += 1
                continue

            elif line.startswith('EDGE_WEIGHT_SECTION'):
                i += 1
                edge_lines = []
                while i < len(lines) and not lines[i].startswith('ITEMS_SECTION') and not lines[i].startswith('NODE_COORD_SECTION'):
                    edge_lines.append(lines[i])
                    i += 1
                
                # Check if lines have triplets (origem, destino, distancia) or matrix format
                first_valid_line = next((l for l in edge_lines if l and not l.startswith('(')), None)
                first_parts = first_valid_line.split() if first_valid_line else []
                
                if len(edge_lines) > 0 and len(first_parts) == 3 and not (dimension > 0 and len(first_parts) == dimension):
                    # Triplet format: u v dist
                    for l in edge_lines:
                        parts = l.split()
                        if len(parts) >= 3:
                            try:
                                u, v, d = int(parts[0]), int(parts[1]), float(parts[2])
                                distancias[(min(u, v), max(u, v))] = d
                                if u not in cidades:
                                    cidades[u] = Cidade(id=u)
                                if v not in cidades:
                                    cidades[v] = Cidade(id=v)
                            except ValueError:
                                pass
                else:
                    # Matrix format
                    row = 1
                    col = 1
                    for l in edge_lines:
                        parts = l.split()
                        for part in parts:
                            try:
                                dist = float(part)
                                if row != col:
                                    key = (min(row, col), max(row, col))
                                    distancias[key] = dist
                                col += 1
                                if dimension > 0 and col > dimension:
                                    row += 1
                                    col = 1
                            except ValueError:
                                continue

                for cid in range(1, dimension + 1):
                    if cid not in cidades:
                        cidades[cid] = Cidade(id=cid)
                continue

            elif line.startswith('ITEMS_SECTION'):
                i += 1
                while i < len(lines):
                    parts = lines[i].split()
                    if len(parts) >= 4 and parts[0].isdigit():
                        try:
                            item_id = int(parts[0])
                            profit = float(parts[1])
                            weight = float(parts[2])
                            assigned_node = int(parts[3])
                            item = Item(id=item_id, peso=weight, valor=profit, cidade_origem=assigned_node)
                            itens.append(item)
                            if assigned_node in cidades:
                                cidades[assigned_node].itens.append(item)
                        except ValueError:
                            pass
                    i += 1
                continue

            i += 1

        grafo = Grafo(cidades=cidades, edge_weight_type=edge_weight_type, distancias=distancias)

        return TTPInstance(
            nome=nome,
            grafo=grafo,
            itens=itens,
            capacidade_mochila=capacidade,
            v_min=v_min,
            v_max=v_max,
            taxa_aluguel=taxa,
            constante_degradacao=degradacao,
            dimension=dimension if dimension > 0 else len(cidades),
            num_items=num_items if num_items > 0 else len(itens)
        )

    def _parse_directory(self, dir_path: str) -> TTPInstance:
        # Expected files: parametros_globais.txt, mapa_cidade.txt, definicao_objetos.txt, distribuicao_objetos.txt
        params = {}
        cidades = {}
        itens = []
        edge_weight_type = 'EUC_2D'
        distancias = {}

        # 1. parametros_globais.txt
        param_file = os.path.join(dir_path, 'parametros_globais.txt')
        if os.path.exists(param_file):
            with open(param_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if ':' in line:
                        key, val = [p.strip() for p in line.split(':', 1)]
                        params[key.upper()] = val

        dimension = int(params.get('DIMENSION', 0))
        num_items = int(params.get('NUMBER OF ITEMS', 0))
        capacidade = float(params.get('CAPACITY OF KNAPSACK', params.get('CAPACIDADE_MOCHILA', 0.0)))
        v_min = float(params.get('MIN SPEED', params.get('V_MIN', 0.1)))
        v_max = float(params.get('MAX SPEED', params.get('V_MAX', 1.0)))
        taxa = float(params.get('RENTING RATIO', params.get('TAXA_ALUGUEL_R', 1.0)))
        degradacao_val = (
            params.get('DEGRADATION CONSTANT') or 
            params.get('DEGRADATION') or 
            params.get('CONSTANTE_DEGRADACAO_C') or 
            '1.0'
        )
        degradacao = float(degradacao_val)
        nome = params.get('PROBLEM NAME', os.path.basename(os.path.abspath(dir_path)))
        edge_weight_type = params.get('EDGE_WEIGHT_TYPE', 'EUC_2D').upper()

        # 2. mapa_cidade.txt
        mapa_file = os.path.join(dir_path, 'mapa_cidade.txt')
        if os.path.exists(mapa_file):
            with open(mapa_file, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]

            # Check if headers exist inside mapa_cidade.txt
            content_lines = []
            for l in lines:
                if ':' in l:
                    k, v = [p.strip() for p in l.split(':', 1)]
                    if k.upper() == 'EDGE_WEIGHT_TYPE':
                        edge_weight_type = v.upper()
                else:
                    content_lines.append(l)

            if edge_weight_type == 'EUC_2D':
                for line in content_lines:
                    parts = line.split()
                    if len(parts) >= 3 and parts[0].isdigit():
                        c_id = int(parts[0])
                        x = float(parts[1])
                        y = float(parts[2])
                        cidades[c_id] = Cidade(id=c_id, x=x, y=y)
            else:  # EXPLICIT
                for line in content_lines:
                    parts = line.split()
                    if len(parts) >= 3 and parts[0].isdigit():
                        try:
                            u, v, d = int(parts[0]), int(parts[1]), float(parts[2])
                            distancias[(min(u, v), max(u, v))] = d
                            if u not in cidades:
                                cidades[u] = Cidade(id=u)
                            if v not in cidades:
                                cidades[v] = Cidade(id=v)
                        except ValueError:
                            pass
                for cid in range(1, dimension + 1):
                    if cid not in cidades:
                        cidades[cid] = Cidade(id=cid)

        # 3. definicao_objetos.txt
        itens_dict = {}
        def_file = os.path.join(dir_path, 'definicao_objetos.txt')
        if os.path.exists(def_file):
            with open(def_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3 and parts[0].isdigit():
                        try:
                            item_id = int(parts[0])
                            profit = float(parts[1])
                            weight = float(parts[2])
                            itens_dict[item_id] = {'profit': profit, 'weight': weight}
                        except ValueError:
                            pass

        # 4. distribuicao_objetos.txt
        dist_file = os.path.join(dir_path, 'distribuicao_objetos.txt')
        if os.path.exists(dist_file):
            with open(dist_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2 and parts[0].isdigit():
                        try:
                            item_id = int(parts[0])
                            assigned_node = int(parts[1])
                            if item_id in itens_dict:
                                item = Item(id=item_id, peso=itens_dict[item_id]['weight'], valor=itens_dict[item_id]['profit'], cidade_origem=assigned_node)
                                itens.append(item)
                                if assigned_node in cidades:
                                    cidades[assigned_node].itens.append(item)
                        except ValueError:
                            pass

        grafo = Grafo(cidades=cidades, edge_weight_type=edge_weight_type, distancias=distancias)

        return TTPInstance(
            nome=nome,
            grafo=grafo,
            itens=itens,
            capacidade_mochila=capacidade,
            v_min=v_min,
            v_max=v_max,
            taxa_aluguel=taxa,
            constante_degradacao=degradacao,
            dimension=dimension if dimension > 0 else len(cidades),
            num_items=num_items if num_items > 0 else len(itens)
        )
