#!/usr/bin/env python3
import psycopg2
import numpy as np
import random
from typing import List, Tuple
import time

DB_NAME = "pydantic_ai_rag"
DB_USER = "admin"
DB_PASS = "data@Spher3"
DB_HOST = "127.0.0.1"
DB_PORT = "5432"

class PgVectorFuzzer:
    def __init__(self, dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT):
        self.conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=port)
        self.conn.autocommit = True
        self.cur = self.conn.cursor()
        self.setup_extension()
        
    def setup_extension(self):
        """Setup pgvector extension if not exists"""
        try:
            self.cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        except Exception as e:
            print(f"Failed to create extension: {e}")
            raise

    def generate_random_vector(self, dim: int) -> List[float]:
        """Generate a random vector with given dimension"""
        return list(np.random.uniform(-10, 10, dim))

    def generate_sparse_vector(self, dim: int, sparsity: float = 0.8) -> List[float]:
        """Generate a sparse vector with given dimension and sparsity"""
        vec = self.generate_random_vector(dim)
        zero_positions = random.sample(range(dim), int(dim * sparsity))
        for pos in zero_positions:
            vec[pos] = 0
        return vec

    def vector_to_sql(self, vec: List[float]) -> str:
        """Convert vector to SQL string format"""
        data =  f"'[{','.join(map(str, vec))}]'"
        return data

    def test_vector_functions(self, iterations: int = 100):
        """Test various vector functions"""
        functions_to_test = [
            ("l2_distance", 2), 
            ("inner_product", 2),
            ("cosine_distance", 2),
            ("l1_distance", 2),
            ("vector_dims", 1),
            ("vector_norm", 1),
            ("l2_normalize", 1),
            ("vector_add", 2),
            ("vector_sub", 2),
            ("vector_mul", 2),
            ("vector_concat", 2)
        ]

        for i in range(iterations):
            dim = random.randint(2, 1024)  # Test different dimensions
            vec1 = self.generate_random_vector(dim)
            vec2 = self.generate_random_vector(dim)
            
            for func_name, num_args in functions_to_test:
                try:
                    if num_args == 1:
                        query = f"SELECT {func_name}({self.vector_to_sql(vec1)}::vector)"
                    else:
                        query = f"SELECT {func_name}({self.vector_to_sql(vec1)}::vector, {self.vector_to_sql(vec2)}::vector)"
                    
                    self.cur.execute(query)
                    result = self.cur.fetchone()
                    print(f"Test {i}, {func_name}: Success")
                except Exception as e:
                    print(f"Test {i}, {func_name}: Failed - {str(e)}")

    def test_vector_operators(self, iterations: int = 100):
        """Test vector comparison operators and distance operators"""
        comparison_operators = ['<', '<=', '=', '>=', '>', '<>']
        distance_operators = ['<->', '<#>', '<=>', '<+>']  # L2, Inner Product, Cosine, L1
        arithmetic_operators = ['+', '-', '*', '||']  # Add, Sub, Multiply, Concat
        
        for i in range(iterations):
            dim = random.randint(2, 1024)
            vec1 = self.generate_random_vector(dim)
            vec2 = self.generate_random_vector(dim)
            
            # Test comparison operators
            for op in comparison_operators:
                try:
                    query = f"SELECT {self.vector_to_sql(vec1)}::vector {op} {self.vector_to_sql(vec2)}::vector"
                    self.cur.execute(query)
                    result = self.cur.fetchone()
                    print(f"Test {i}, comparison operator {op}: Success")
                except Exception as e:
                    print(f"Test {i}, comparison operator {op}: Failed - {str(e)}")
            
            # Test distance operators
            for op in distance_operators:
                try:
                    query = f"SELECT {self.vector_to_sql(vec1)}::vector {op} {self.vector_to_sql(vec2)}::vector"
                    self.cur.execute(query)
                    result = self.cur.fetchone()
                    print(f"Test {i}, distance operator {op}: Success")
                except Exception as e:
                    print(f"Test {i}, distance operator {op}: Failed - {str(e)}")
            
            # Test arithmetic operators
            for op in arithmetic_operators:
                try:
                    query = f"SELECT {self.vector_to_sql(vec1)}::vector {op} {self.vector_to_sql(vec2)}::vector"
                    self.cur.execute(query)
                    result = self.cur.fetchone()
                    print(f"Test {i}, arithmetic operator {op}: Success")
                except Exception as e:
                    print(f"Test {i}, arithmetic operator {op}: Failed - {str(e)}")

    def test_halfvec_functions(self, iterations: int = 100):
        """Test halfvec type functions"""
        functions_to_test = [
            ("l2_distance", 2),
            ("inner_product", 2),
            ("cosine_distance", 2),
            ("l1_distance", 2),
            ("vector_dims", 1),
            ("l2_norm", 1),
            ("l2_normalize", 1),
            ("binary_quantize", 1)
        ]
        
        for i in range(iterations):
            dim = random.randint(2, 1024)
            vec1 = self.generate_random_vector(dim)
            vec2 = self.generate_random_vector(dim)
            
            for func_name, num_args in functions_to_test:
                try:
                    if num_args == 1:
                        query = f"SELECT {func_name}({self.vector_to_sql(vec1)}::halfvec)"
                    else:
                        query = f"SELECT {func_name}({self.vector_to_sql(vec1)}::halfvec, {self.vector_to_sql(vec2)}::halfvec)"
                    
                    self.cur.execute(query)
                    result = self.cur.fetchone()
                    print(f"Test {i}, halfvec {func_name}: Success")
                except Exception as e:
                    print(f"Test {i}, halfvec {func_name}: Failed - {str(e)}")

    def test_halfvec_operators(self, iterations: int = 100):
        """Test halfvec operators including distance, arithmetic, and comparison operators"""
        distance_operators = ['<->', '<#>', '<=>', '<+>']  # L2, Inner Product, Cosine, L1
        arithmetic_operators = ['+', '-', '*', '||']  # Add, Sub, Multiply, Concat
        comparison_operators = ['<', '<=', '=', '>=', '>', '<>']
        
        for i in range(iterations):
            dim = random.randint(2, 1024)
            vec1 = self.generate_random_vector(dim)
            vec2 = self.generate_random_vector(dim)
            
            # Test distance operators
            for op in distance_operators:
                try:
                    query = f"SELECT {self.vector_to_sql(vec1)}::halfvec {op} {self.vector_to_sql(vec2)}::halfvec"
                    self.cur.execute(query)
                    result = self.cur.fetchone()
                    print(f"Test {i}, halfvec distance operator {op}: Success")
                except Exception as e:
                    print(f"Test {i}, halfvec distance operator {op}: Failed - {str(e)}")
            
            # Test arithmetic operators
            for op in arithmetic_operators:
                try:
                    query = f"SELECT {self.vector_to_sql(vec1)}::halfvec {op} {self.vector_to_sql(vec2)}::halfvec"
                    self.cur.execute(query)
                    result = self.cur.fetchone()
                    print(f"Test {i}, halfvec arithmetic operator {op}: Success")
                except Exception as e:
                    print(f"Test {i}, halfvec arithmetic operator {op}: Failed - {str(e)}")
            
            # Test comparison operators
            for op in comparison_operators:
                try:
                    query = f"SELECT {self.vector_to_sql(vec1)}::halfvec {op} {self.vector_to_sql(vec2)}::halfvec"
                    self.cur.execute(query)
                    result = self.cur.fetchone()
                    print(f"Test {i}, halfvec comparison operator {op}: Success")
                except Exception as e:
                    print(f"Test {i}, halfvec comparison operator {op}: Failed - {str(e)}")

    def test_halfvec_aggregates(self):
        """Test halfvec aggregate functions (avg, sum)"""
        try:
            # Create test table
            self.cur.execute("DROP TABLE IF EXISTS halfvec_test")
            self.cur.execute("CREATE TABLE halfvec_test (id serial primary key, embedding halfvec(3))")
            
            # Insert test data
            test_vectors = [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0]
            ]
            for vec in test_vectors:
                self.cur.execute(
                    f"INSERT INTO halfvec_test (embedding) VALUES ('{self.vector_to_sql(vec).replace("'", '')}'::halfvec)"
                )
            
            # Test AVG aggregate
            try:
                self.cur.execute("SELECT avg(embedding) FROM halfvec_test")
                result = self.cur.fetchone()
                print("Halfvec AVG aggregate: Success")
            except Exception as e:
                print(f"Halfvec AVG aggregate: Failed - {str(e)}")
            
            # Test SUM aggregate
            try:
                self.cur.execute("SELECT sum(embedding) FROM halfvec_test")
                result = self.cur.fetchone()
                print("Halfvec SUM aggregate: Success")
            except Exception as e:
                print(f"Halfvec SUM aggregate: Failed - {str(e)}")
            
        finally:
            # Cleanup
            self.cur.execute("DROP TABLE IF EXISTS halfvec_test")

    def test_halfvec_float_conversion(self, iterations: int = 100):
        """Test halfvec to float array conversion"""
        for i in range(iterations):
            dim = random.randint(2, 16)  # Smaller dimension for conversion tests
            vec = self.generate_random_vector(dim)
            
            try:
                query = f"SELECT {self.vector_to_sql(vec)}::halfvec::real[]"
                self.cur.execute(query)
                result = self.cur.fetchone()
                print(f"Test {i}, halfvec to float array conversion: Success")
            except Exception as e:
                print(f"Test {i}, halfvec to float array conversion: Failed - {str(e)}")

    def test_sparsevec_functions(self, iterations: int = 100):
        """Test sparsevec type functions"""
        functions_to_test = [
            ("l2_distance", 2),
            ("inner_product", 2),
            ("cosine_distance", 2),
            ("l1_distance", 2),
            ("l2_norm", 1)
        ]
        
        for i in range(iterations):
            dim = random.randint(2, 1024)
            vec1 = self.generate_sparse_vector(dim)
            vec2 = self.generate_sparse_vector(dim)
            
            # Convert to sparsevec format
            def to_sparsevec(vec):
                nonzero = [(i+1, v) for i, v in enumerate(vec) if v != 0]
                if not nonzero:
                    return "'{}/0'"
                indices, values = zip(*nonzero)
                return f"'{','.join(f'{i}:{v}' for i, v in zip(indices, values))}/{max(indices)}'"
            
            for func_name, num_args in functions_to_test:
                try:
                    if num_args == 1:
                        query = f"SELECT {func_name}({to_sparsevec(vec1)}::sparsevec)"
                    else:
                        query = f"SELECT {func_name}({to_sparsevec(vec1)}::sparsevec, {to_sparsevec(vec2)}::sparsevec)"
                    
                    self.cur.execute(query)
                    result = self.cur.fetchone()
                    print(f"Test {i}, sparsevec {func_name}: Success")
                except Exception as e:
                    print(f"Test {i}, sparsevec {func_name}: Failed - {str(e)}")

    def test_edge_cases(self):
        """Test edge cases"""
        edge_cases = [
            ([0] * 10),  # Zero vector
            ([1e10] * 10),  # Very large values
            ([1e-10] * 10),  # Very small values
            ([float('nan')] * 10),  # NaN values
            ([float('inf')] * 10),  # Infinity
        ]
        
        for vec in edge_cases:
            try:
                query = f"SELECT vector_norm({self.vector_to_sql(vec)}::vector)"
                self.cur.execute(query)
                result = self.cur.fetchone()
                print(f"Edge case {vec[:3]}...: Success")
            except Exception as e:
                print(f"Edge case {vec[:3]}...: Failed - {str(e)}")

    def test_binary_quantization(self, iterations: int = 100):
        """Test binary quantization"""
        for i in range(iterations):
            dim = random.randint(2, 1024)
            vec = self.generate_random_vector(dim)
            
            try:
                query = f"SELECT binary_quantize({self.vector_to_sql(vec)}::vector)"
                self.cur.execute(query)
                result = self.cur.fetchone()
                print(f"Binary quantization test {i}: Success")
            except Exception as e:
                print(f"Binary quantization test {i}: Failed - {str(e)}")

    def test_type_casts(self, iterations: int = 100):
        """Test type casting between vector, halfvec, and sparsevec"""
        casts = [
            ("vector", "sparsevec"),
            ("sparsevec", "vector"),
            ("sparsevec", "halfvec"),
            ("halfvec", "sparsevec")
        ]
        
        for i in range(iterations):
            dim = random.randint(2, 1024)
            vec = self.generate_random_vector(dim)
            
            for source_type, target_type in casts:
                try:
                    # First cast to source type, then to target type
                    query = f"SELECT {self.vector_to_sql(vec)}::{source_type}::{target_type}"
                    self.cur.execute(query)
                    result = self.cur.fetchone()
                    print(f"Test {i}, cast {source_type}->{target_type}: Success")
                except Exception as e:
                    print(f"Test {i}, cast {source_type}->{target_type}: Failed - {str(e)}")

    def test_array_conversions(self, iterations: int = 100):
        """Test array to vector type conversions"""
        array_types = ["integer[]", "real[]", "double precision[]", "numeric[]"]
        vector_types = ["vector", "halfvec", "sparsevec"]
        
        for i in range(iterations):
            dim = random.randint(2, 16)  # Smaller dimension for array tests
            arr = [random.randint(-100, 100) for _ in range(dim)]
            
            for array_type in array_types:
                for vector_type in vector_types:
                    try:
                        # Convert array to vector type
                        query = f"SELECT ARRAY{arr}::{array_type}::{vector_type}"
                        self.cur.execute(query)
                        result = self.cur.fetchone()
                        print(f"Test {i}, convert {array_type}->{vector_type}: Success")
                    except Exception as e:
                        print(f"Test {i}, convert {array_type}->{vector_type}: Failed - {str(e)}")

    def test_index_operations(self):
        """Test HNSW and IVFFlat index operations"""
        try:
            # Create test table
            self.cur.execute("DROP TABLE IF EXISTS vector_test")
            self.cur.execute("CREATE TABLE vector_test (id serial primary key, embedding vector(3))")
            
            # Insert test data
            test_vectors = [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0]
            ]
            for vec in test_vectors:
                query = f"INSERT INTO vector_test (embedding) VALUES  ({self.vector_to_sql(vec)})"
                self.cur.execute(query)

            # Test different index types and operators
            index_configs = [
                ("ivfflat", "vector_l2_ops", "<->"),
                ("ivfflat", "vector_ip_ops", "<#>"),
                ("ivfflat", "vector_cosine_ops", "<=>"),
                ("hnsw", "vector_l2_ops", "<->"),
                ("hnsw", "vector_ip_ops", "<#>"),
                ("hnsw", "vector_cosine_ops", "<=>"),
                ("hnsw", "vector_l1_ops", "<+>")
            ]
            
            for index_method, opclass, operator in index_configs:
                try:
                    # Create index
                    index_name = f"idx_{index_method}_{opclass}"
                    self.cur.execute(f"DROP INDEX IF EXISTS {index_name}")
                    
                    if index_method == "ivfflat":
                        self.cur.execute(
                            f"CREATE INDEX {index_name} ON vector_test USING {index_method} "
                            f"(embedding {opclass}) WITH (lists = 2)"
                        )
                    else:  # hnsw
                        self.cur.execute(
                            f"CREATE INDEX {index_name} ON vector_test USING {index_method} "
                            f"(embedding {opclass}) WITH (m = 2, ef_construction = 10)"
                        )
                    
                    # Test query
                    query_vec = [0.5, 0.5, 0.5]

                    query = f"SELECT id FROM vector_test ORDER BY embedding {operator} {self.vector_to_sql(query_vec)}::vector LIMIT 1"

                    self.cur.execute(
                        query
                    )
                    result = self.cur.fetchone()
                    print(f"Index test {index_method} {opclass}: Success")
                    
                except Exception as e:
                    print(f"Index test {index_method} {opclass}: Failed - {str(e)}")
                    
        finally:
            # Cleanup
            self.cur.execute("DROP TABLE IF EXISTS vector_test")

    def generate_random_bit_vector(self, length: int) -> str:
        """Generate a random bit vector string"""
        return ''.join(random.choice(['0', '1']) for _ in range(length))

    def test_bit_operations(self, iterations: int = 100):
        """Test bit vector operations (Hamming and Jaccard distance)"""
        operators = {
            '<~>': 'hamming_distance',
            '<%>': 'jaccard_distance'
        }
        
        for i in range(iterations):
            # Test with different bit lengths
            length = random.randint(8, 64)
            bit1 = self.generate_random_bit_vector(length)
            bit2 = self.generate_random_bit_vector(length)
            
            # Test operators
            for op, func_name in operators.items():
                try:
                    # Test operator syntax
                    query = f"SELECT B'{bit1}' {op} B'{bit2}'"
                    self.cur.execute(query)
                    result = self.cur.fetchone()
                    print(f"Test {i}, bit operator {op}: Success")
                    
                    # Test function syntax
                    query = f"SELECT {func_name}(B'{bit1}', B'{bit2}')"
                    self.cur.execute(query)
                    result = self.cur.fetchone()
                    print(f"Test {i}, bit function {func_name}: Success")
                except Exception as e:
                    print(f"Test {i}, bit operation {op}/{func_name}: Failed - {str(e)}")

    def test_halfvec_index_operations(self):
        """Test halfvec index operations with HNSW and IVFFlat"""
        try:
            # Create test table
            self.cur.execute("DROP TABLE IF EXISTS halfvec_index_test")
            self.cur.execute("CREATE TABLE halfvec_index_test (id serial primary key, embedding halfvec(3))")
            
            # Insert test data
            test_vectors = [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0]
            ]
            for vec in test_vectors:
                query = f"INSERT INTO halfvec_index_test (embedding) VALUES ({self.vector_to_sql(vec)}::halfvec)"
                print(query)
                self.cur.execute(query)
            
            # Test different index types and operators
            index_configs = [
                ("ivfflat", "halfvec_l2_ops", "<->"),
                ("ivfflat", "halfvec_ip_ops", "<#>"),
                ("ivfflat", "halfvec_cosine_ops", "<=>"),
                ("hnsw", "halfvec_l2_ops", "<->"),
                ("hnsw", "halfvec_ip_ops", "<#>"),
                ("hnsw", "halfvec_cosine_ops", "<=>"),
                ("hnsw", "halfvec_l1_ops", "<+>")
            ]
            
            for index_method, opclass, operator in index_configs:
                try:
                    # Create index
                    index_name = f"idx_halfvec_{index_method}_{opclass}"
                    self.cur.execute(f"DROP INDEX IF EXISTS {index_name}")
                    
                    if index_method == "ivfflat":
                        self.cur.execute(
                            f"CREATE INDEX {index_name} ON halfvec_index_test USING {index_method} "
                            f"(embedding {opclass}) WITH (lists = 2)"
                        )
                    else:  # hnsw
                        self.cur.execute(
                            f"CREATE INDEX {index_name} ON halfvec_index_test USING {index_method} "
                            f"(embedding {opclass}) WITH (m = 2, ef_construction = 10)"
                        )
                    
                    # Test query
                    query_vec = [0.5, 0.5, 0.5]
                    self.cur.execute(
                        f"SELECT id FROM halfvec_index_test ORDER BY embedding {operator} %s::halfvec LIMIT 1",
                        (self.vector_to_sql(query_vec),)
                    )
                    result = self.cur.fetchone()
                    print(f"Halfvec index test {index_method} {opclass}: Success")
                    
                except Exception as e:
                    print(f"Halfvec index test {index_method} {opclass}: Failed - {str(e)}")
                    
        finally:
            # Cleanup
            self.cur.execute("DROP TABLE IF EXISTS halfvec_index_test")

    def test_bit_index_operations(self):
        """Test bit vector index operations"""
        try:
            # Create test table
            self.cur.execute("DROP TABLE IF EXISTS bit_test")
            self.cur.execute("CREATE TABLE bit_test (id serial primary key, bits bit(32))")
            
            # Insert test data
            test_bits = [
                '10101010101010101010101010101010',
                '11111111000000001111111100000000',
                '00000000111111110000000011111111',
                '11111111111111110000000000000000'
            ]
            for bits in test_bits:
                self.cur.execute(
                    "INSERT INTO bit_test (bits) VALUES (B%s)",
                    (bits,)
                )
            
            # Test different index types and operators
            index_configs = [
                ("ivfflat", "bit_hamming_ops", "<~>"),
                ("hnsw", "bit_hamming_ops", "<~>"),
                ("hnsw", "bit_jaccard_ops", "<%>")
            ]
            
            for index_method, opclass, operator in index_configs:
                try:
                    # Create index
                    index_name = f"idx_bit_{index_method}_{opclass}"
                    self.cur.execute(f"DROP INDEX IF EXISTS {index_name}")
                    
                    if index_method == "ivfflat":
                        self.cur.execute(
                            f"CREATE INDEX {index_name} ON bit_test USING {index_method} "
                            f"(bits {opclass}) WITH (lists = 2)"
                        )
                    else:  # hnsw
                        self.cur.execute(
                            f"CREATE INDEX {index_name} ON bit_test USING {index_method} "
                            f"(bits {opclass}) WITH (m = 2, ef_construction = 10)"
                        )
                    
                    # Test query
                    query_bits = '10101010101010101010101010101010'
                    self.cur.execute(
                        f"SELECT id FROM bit_test ORDER BY bits {operator} B'{query_bits}' LIMIT 1"
                    )
                    result = self.cur.fetchone()
                    print(f"Bit index test {index_method} {opclass}: Success")
                    
                except Exception as e:
                    print(f"Bit index test {index_method} {opclass}: Failed - {str(e)}")
                    
        finally:
            # Cleanup
            self.cur.execute("DROP TABLE IF EXISTS bit_test")

    def run_all_tests(self, iterations: int = 100):
        """Run all fuzzing tests"""
        print("Starting pgvector fuzzing...")
        
        print("\n=== Testing Vector Functions ===")
        self.test_vector_functions(iterations)
        
        print("\n=== Testing Vector Operators ===")
        self.test_vector_operators(iterations)
        
        print("\n=== Testing Edge Cases ===")
        self.test_edge_cases()
        
        print("\n=== Testing Binary Quantization ===")
        self.test_binary_quantization(iterations)

        print("\n=== Testing Halfvec Functions ===")
        self.test_halfvec_functions(iterations)

        print("\n=== Testing Halfvec Operators ===")
        self.test_halfvec_operators(iterations)

        print("\n=== Testing Halfvec Aggregates ===")
        self.test_halfvec_aggregates()

        print("\n=== Testing Halfvec Float Conversion ===")
        self.test_halfvec_float_conversion(iterations)

        print("\n=== Testing Halfvec Index Operations ===")
        self.test_halfvec_index_operations()

        print("\n=== Testing Bit Operations ===")
        self.test_bit_operations(iterations)

        print("\n=== Testing Bit Index Operations ===")
        self.test_bit_index_operations()

        print("\n=== Testing Sparsevec Functions ===")
        self.test_sparsevec_functions(iterations)

        print("\n=== Testing Type Casts ===")
        self.test_type_casts(iterations)

        print("\n=== Testing Array Conversions ===")
        self.test_array_conversions(iterations)

        print("\n=== Testing Index Operations ===")
        self.test_index_operations()

if __name__ == "__main__":
    fuzzer = PgVectorFuzzer()
    fuzzer.run_all_tests()
