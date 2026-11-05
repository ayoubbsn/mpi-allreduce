pipeline {
    agent {
        dockerfile {
            filename 'Dockerfile'
        }
    }

    options {
        timeout(time: 30, unit: 'MINUTES')
    }

    stages {
        stage('Build') {
            steps {
                sh 'cmake -S . -B build -DCMAKE_BUILD_TYPE=Release'
                sh 'cmake --build build -j'
            }
        }

        stage('Test') {
            steps {
                sh 'ctest --test-dir build --output-on-failure --output-junit ctest.xml'
            }
            post {
                always {
                    junit 'build/ctest.xml'
                }
            }
        }

        stage('Correctness np=4') {
            steps {
                sh 'bash tests/run_tests.sh build/test_correctness'
            }
        }

        stage('Benchmark') {
            steps {
                sh 'mpirun -np 4 --oversubscribe build/bench -m 1M -o bench.csv'
                sh 'python3 scripts/plot_bench.py bench.csv'
            }
            post {
                success {
                    archiveArtifacts artifacts: 'bench.csv, bench.png'
                }
            }
        }
    }
}
