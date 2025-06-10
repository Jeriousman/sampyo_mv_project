#!/bin/bash

systemctl stop server_socket.service
sleep 300 
systemctl restart server_socket.service
