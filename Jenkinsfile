/* Copyright 2024 National Research Foundation (SARAO)
 *
 * This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/* This Jenkinsfile is specific to the environment at SARAO, and is intended
 * mainly to allow testing the ibverbs support against mlx5 hardware. It can
 * probably be made to work elsewhere, but you shouldn't expect it to work
 * out of the box. Note that the bulk of spead2 is tested on Github Actions
 * (see .github/workflows).
 */

pipeline {
  agent {
    docker {
      label 'spead2'
      registryCredentialsId 'dockerhub'  // Supply credentials to avoid rate limit
      image 'ubuntu:24.04'
      // These arguments allow direct access to the NVIDIA NIC with ibverbs
      args '--network=host --ulimit=memlock=-1 -e NVIDIA_MOFED=enabled -e NVIDIA_VISIBLE_DEVICES=all --runtime=nvidia'
    }
  }

  options {
    timeout(time: 30, unit: 'MINUTES')
  }

  environment {
    DEBIAN_FRONTEND = 'noninteractive'
    // Setting PATH here doesn't seem to work
  }

  stages {
    stage('Install dependencies') {
      steps {
        // jq and iproute2 are used to find the interface IPv4 address
        sh 'apt-get update && apt-get -y --no-install-recommends install libpython3-dev python3-venv jq iproute2'
        sh 'python3 -m venv /venv'
        sh '.ci/install-sys-pkgs.sh'
        sh 'PATH="/venv/bin:$PATH" .ci/py-requirements.sh'
      }
    }

    stage('Install Python package') {
      steps {
        sh '''PATH="/venv/bin:$PATH" pip install -v \
          --config-settings=setup-args=--native-file=ci.ini \
          --config-settings=setup-args=-Dibv=enabled \
          --config-settings=setup-args=-Dibv_hw_rate_limit=enabled \
          --config-settings=setup-args=-Dmlx5dv=enabled \
          .'''
      }
    }

    stage('Run tests') {
      steps {
        sh 'PATH="/venv/bin:$PATH" .ci/py-tests-jenkins.sh'
        junit 'results.xml'
        sh 'PATH="/venv/bin:$PATH" .ci/py-tests-shutdown.sh'
      }
    }
  }

  post {
    always {
      emailext(
        attachLog: true,
        to: '$DEFAULT_RECIPIENTS',
        subject: '$PROJECT_NAME - $BUILD_STATUS',
        body: '${SCRIPT, template="groovy-html.template"}',
      )
      cleanWs()
    }
  }
}
